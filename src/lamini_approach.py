from azure.search.documents import SearchClient
from azure.search.documents.indexes.models import (
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    HnswParameters,
    VectorSearchAlgorithmMetric,
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    VectorSearchProfile,
    AzureOpenAIVectorizer,
    AzureOpenAIParameters,
    SemanticConfiguration,
    SemanticSearch,
    SemanticPrioritizedFields,
    SemanticField,
    SearchIndex
)
from azure.search.documents.indexes import SearchIndexClient
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
import os
import ast
import streamlit as st
from tenacity import retry, wait_random_exponential, stop_after_attempt
import uuid
from dotenv import load_dotenv

load_dotenv(".env")

from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryType,
    VectorizedQuery,
)


def similarity_search(query, database_name='quickinsight', cache_threshold=0.94):
    cached_sql_query = ''
    vector_query = VectorizedQuery(vector=generate_embeddings(
        query), k_nearest_neighbors=3, fields="user_query_vector")

    results = search_client.search(
        search_text="",
        vector_queries=[vector_query],
        filter=f"database_name eq '{database_name}' and sql_query ne 'NA'",
        select=["user_query", "sql_query", "database_name"],
        top=3)

    results = list(results)
    if results:
        results_ = results[0]
        if results_['@search.score'] >= cache_threshold:
            cached_sql_query = results_['sql_query']

    # results = [x for x in results]

    return results, cached_sql_query


azure_search_service_endpoint = st.secrets["AZURE_SEARCH_SERVICE_ENDPOINT"]
azure_search_admin_key = st.secrets["AZURE_SEARCH_SERVICE_KEY"]
credential = AzureKeyCredential(azure_search_admin_key)
index_name = st.secrets["AZURE_INDEX"] # os.environ["index_name"]
azure_openai_endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"]
azure_openai_key = st.secrets["AZURE_OPENAI_KEY"]
azure_openai_api_version = st.secrets["AZURE_OPENAI_VERSION"]
azure_openai_gpt_model_deployment_name = st.secrets["AZURE_OPENAI_MODEL"]
azure_openai_embedding_deployment_name = st.secrets["AZURE_EMBEDDING_MODEL"]







client = AzureOpenAI(
    api_key=azure_openai_key,
    api_version=azure_openai_api_version,
    azure_endpoint=azure_openai_endpoint
)


def openai_chat(user_query, sql_query, number_of_queries=5):
    template = """Generate {number_of_queries} similar user queries based on the given user query and sql query below. Ensure that the generated queries maintain the same context .Each user query should be clearly stated. The newly generated user queries should result in same SQL query given below

    — User Query—
    {user_query}
    — SQL Query—
    {sql_query}

    Give answer in the following Response format:["query1","query2","query3",...,"queryN"]
    """

    prompt = template.format(
        user_query=user_query, sql_query=sql_query, number_of_queries=number_of_queries)

    response = client.chat.completions.create(
        model=azure_openai_gpt_model_deployment_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{prompt}"}
        ],
        max_tokens=1024,
        n=1,
        temperature=0.7,
    )

    answer = response.choices[0].message.content
    answer = ast.literal_eval(answer)
    # print(f"Question: {prompt}\n")
    # print(f"Answer:\n{answer}\n")
    return answer


# answer = openai_chat(user_query, sql_query)
# answer

# Convert the string to an actual list
# query_list = ast.literal_eval(answer)
# query_list
# query_list.append(user_query)


def create_update_search_index():
    # Create a search index
    index_client = SearchIndexClient(
        endpoint=azure_search_service_endpoint, credential=credential)
    fields = [
        SearchField(name="id", type=SearchFieldDataType.String,
                    sortable=True, filterable=True, facetable=True, key=True),
        SearchField(name="database_name",
                    type=SearchFieldDataType.String, filterable=True),
        SearchField(name="user_query", type=SearchFieldDataType.String,
                    sortable=True, filterable=True, facetable=True),
        SearchField(name="sql_query", type=SearchFieldDataType.String,
                    sortable=True, filterable=True, facetable=True),
        SearchField(name="user_query_vector", type=SearchFieldDataType.Collection(
            SearchFieldDataType.Single), vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    ]

    # Configure the vector search configuration
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw",
                parameters=HnswParameters(
                    m=4,
                    ef_construction=400,
                    ef_search=500,
                    metric=VectorSearchAlgorithmMetric.COSINE,
                ),
            ),
            ExhaustiveKnnAlgorithmConfiguration(
                name="myExhaustiveKnn",
                parameters=ExhaustiveKnnParameters(
                    metric=VectorSearchAlgorithmMetric.COSINE,
                ),
            ),
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw",
                vectorizer="myOpenAI",
            ),
            VectorSearchProfile(
                name="myExhaustiveKnnProfile",
                algorithm_configuration_name="myExhaustiveKnn",
                vectorizer="myOpenAI",
            ),
        ],
        vectorizers=[
            AzureOpenAIVectorizer(
                name="myOpenAI",
                kind="azureOpenAI",
                azure_open_ai_parameters=AzureOpenAIParameters(
                    resource_uri=azure_openai_endpoint,
                    deployment_id=azure_openai_embedding_deployment_name,
                    api_key=azure_openai_key,
                    # model_name = azure_openai_embedding_model_name,

                ),
            ),
        ],
    )

    semantic_config = SemanticConfiguration(
        name="my-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[SemanticField(field_name="user_query")]
        ),
    )

    # Create the semantic search with the configuration
    semantic_search = SemanticSearch(configurations=[semantic_config])

    # Create the search index
    index = SearchIndex(name=index_name, fields=fields,
                        vector_search=vector_search, semantic_search=semantic_search)
    result = index_client.create_or_update_index(index)
    print(f"{result.name} created")


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
# Function to generate embeddings for title and content fields, also used for query embeddings
def generate_embeddings(text):
    return client.embeddings.create(input=[text], model=azure_openai_embedding_deployment_name).data[0].embedding


search_client = SearchClient(
    endpoint=azure_search_service_endpoint, index_name=index_name, credential=credential)
# Upload some documents to the index


def upload_to_search_index(query_list, sql_query, database_name="quickinsight"):
    documents = []
    # similar_query_id = str(uuid.uuid4())
    if str(sql_query).strip().lower() == 'na':
        sql_query = 'NA'
    for i in query_list:
        item = {}
        item["id"] = str(uuid.uuid4())
        item["database_name"] = database_name
        item["user_query"] = i
        item["sql_query"] = sql_query
        item["user_query_vector"] = generate_embeddings(i)
        # item["similar_query_id"] = similar_query_id
        documents.append(item)
    search_client.upload_documents(documents)
    st.write("Updated memory")
    # st.write(f"Uploaded {len(documents)} documents")
    # print(f"Uploaded {len(documents)} documents")







# if __name__ == "__main__":
#     upload_to_search_index(['name of all the customers who are married males who have made a total purchase amount of atleast 10000'], "SELECT c.CustomerKey,c.FirstName,c.LastName FROM AdventureWorks_Customers c JOIN AdventureWorks_Sales s ON c.CustomerKey = s.CustomerKey JOIN AdventureWorks_Products p ON s.ProductKey = p.ProductKey WHERE c.Gender = 'M' AND c.MaritalStatus = 'M' GROUP BY c.CustomerKey, c.FirstName, c.LastName HAVING SUM(s.OrderQuantity * p.ProductPrice) >= 10000;", 'quickinsight')