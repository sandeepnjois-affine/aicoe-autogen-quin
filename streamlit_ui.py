from websockets.sync.client import connect as ws_connect
import pandas as pd
import autogen
from autogen.io.websockets import IOWebsockets
from src.agents_on_connect_file import on_connect
from src.lamini_approach import *
import streamlit as st
from queue import Queue
import time
import re
from openai import AzureOpenAI
import ast
import sqlparse
import os
import shutil
# Function to handle and display messages
client = AzureOpenAI(
    azure_endpoint='https://aipractices.openai.azure.com/',
    api_version='2023-12-01-preview',
    api_key='1dfa2422e0ba43a88044e87df4655c4c')



def extract_list_of_dicts(input_string):
    try:
        # Use regex to find the list of dictionaries in the string
        match = re.search(r'\[.*?\]', input_string)
        if match:
            list_str = match.group(0)
            # Use literal_eval to safely evaluate the string
            result = ast.literal_eval(list_str)
            return result
            # Ensure the result is a list of dictionaries
            # if isinstance(result, list) and all(isinstance(i, dict) for i in result):
            #     return result
            # else:
            #     raise ValueError("Extracted part is not a valid list of dictionaries.")
        else:
            raise ValueError(
                "No list of dictionaries found in the input string.")
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Invalid input string: {e}")


def summarize_chat_result(chat_result):
    manage_summary_format = [{"sql_query": "(generated_sql_query)", "insights": "(insights)", 'sql_critic_iter': '(Eg. 2)', 'sql_critic_messages': '(sql_critic_messages)',
                              'insights_critic_iter': '(Eg. 1)', 'insights_critic_messages': '(insights_critic_messages)', 'no_of_tokens': '(no_of_tokens)', 'cost': '(cost)'}]
    prompt = f"""



    Step 1: Identify which scenario the chat_result falls under
    Step 2: Follow the instructions of that scenario as mentioned below

    Do it in steps

    There can be any one of the below mentioned 3 scenarios: 

    FIRST SCENARIO: If 'terminator' is NOT present in 'name' key of any dictionary of chat_result.
    1. Analyze the chat_result from the last dictionary where the 'name' key is insights_critic and answer generated_sql_query and insights in this format: {str(manage_summary_format)}. 
    2. Analyze the complete chat_result and answer how many times sql_critic has given Score 0 and insights_critic has given Score 0. If they have not given Score 0 even once, sql_critic_iter and insights_critic_iter as 1. If they have given Score 0 once, then sql_critic_iter and insights_critic_iter as 2. Basically we are tracking at which iteration the critic agent got it right.
    3. Answer all the critic messages given by sql_critic and insights_critic respectively.
    4. Also answer the total no. of tokens consumsed and cost.
    5. Answer in the form of a valid list of dictionary in one line ONLY

    SECOND SCENARIO: If 'terminator' is present in 'name' key of any dictionary of chat_result AND if 'terminator' is after 'sql_critic'
    Answer: sql_query: 'NA', 'insights': 'NA', 'sql_critic_iter': 3, 'sql_critic_messages': '(summaize all sql_critic critic messages)', 'insights_critic_iter': 0, 'insights_critic_messages': 'NA'. no_of_tokens and cost as whatever they are

    THIRD SCENARIO: If 'terminator' is present in 'name' key of any dictionary of chat_result AND if 'terminator' is after 'insights_critic'
    Answer: sql_query: '(generated_sql_query)', 'insights': 'NA', 'sql_critic_iter':'(Eg. 2)' , 'sql_critic_messages': '(sql_critic_messages)', 'insights_critic_iter': 3, 'insights_critic_messages': '(summaize all insights_critic critic messages)'. no_of_tokens and cost as whatever they are




    \nDo NOT add any introductory phrases or other explanation.\n\n


    chat_result -- > {str(chat_result)}"""

    completion = client.chat.completions.create(
        model='gpt-4o-05-13',
        temperature=0,
        messages=[{'role': 'system', 'content': 'You are a helpful assistant who is an expert in analyzing text data and formatting.'},
                  {"role": "user", "content": prompt}])
    output = completion.choices[0].message.content
    # print('output LLM', output)
    output_ = extract_list_of_dicts(output)
    # print('output_ LLM:', output_)
    # sql_query = output_[0]['sql_query']
    # insights = output_[0]['insights']

    return output_


def strip_ansi_codes(text):
    ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)


def handle_message(message):
    # Check for green text
    if "\x1B[32m" in message and "\x1B[0m" in message:
        clean_text = strip_ansi_codes(message)
        print("line 104: ", clean_text)
        st.markdown(
            f'<p style="color: green;">{clean_text}</p>', unsafe_allow_html=True)
    # Check for dark blue text
    elif "\x1B[33m" in message and "\x1B[0m" in message:
        clean_text = strip_ansi_codes(message)
        print("line 110: ", clean_text)
        st.markdown(
            f'<p style="color: darkblue;">{clean_text}</p>', unsafe_allow_html=True)
    # Check for magenta text
    elif "\x1B[35m" in message and "\x1B[0m" in message:
        clean_text = strip_ansi_codes(message)
        print("line 116: ", clean_text)
        st.markdown(
            f'<p style="color: magenta;">{clean_text}</p>', unsafe_allow_html=True)
    # Check for green text without closing sequence
    elif "\x1B[32m" in message and "\x1B[0m" not in message:
        clean_text = strip_ansi_codes(message.replace('\x1B[32m', ''))
        print("line 122: ", clean_text)
        st.markdown(
            f'<p style="color: green;">{clean_text}</p>', unsafe_allow_html=True)
    # Check for dark blue text without closing sequence
    elif "\x1B[33m" in message and "\x1B[0m" not in message:
        clean_text = strip_ansi_codes(message.replace('\x1B[33m', ''))
        print("line 128: ", clean_text)
        st.markdown(
            f'<p style="color: darkblue;">{clean_text}</p>', unsafe_allow_html=True)
    # Check for magenta text without closing sequence
    elif "\x1B[35m" in message and "\x1B[0m" not in message:
        clean_text = strip_ansi_codes(message.replace('\x1B[35m', ''))
        print("line 134: ", clean_text)
        st.markdown(
            f'<p style="color: magenta;">{clean_text}</p>', unsafe_allow_html=True)
    # Check for closing sequence without opening sequence
    elif "\x1B[0m" in message and "\x1B[32m" not in message and "\x1B[33m" not in message and "\x1B[35m" not in message:
        clean_text = strip_ansi_codes(message)
        print("line 140: ", clean_text)
        if ' ' in str(clean_text).strip():
            st.write(clean_text)
        else:
            pass
        # print(clean_text, end="", flush=True)
        # st.markdown(f'<div style="font-size: 18px; font-weight: bold; color: #2e7d32;">{clean_text}</div>', unsafe_allow_html=True)
    # Handle normal text
    else:
        if 'Next speaker:' in str(message).strip():
            st.markdown(
            f'<p style="color: green;">{str(message).strip()}</p>', unsafe_allow_html=True)
        elif '(to chat_manager):' in str(message).strip():
            st.markdown(
            f'<p style="color: darkblue;">{str(message).strip()}</p>', unsafe_allow_html=True)
        elif ' ' in str(message).strip():
            print('line 150:  ', str(message).strip())
            st.write(message)
        else:
            pass
        # print(message, end="", flush=True)

        # st.markdown(f'<div style=
        # st.markdown(f'<div style="font-size: 18px; font-weight: bold; color: #2e7d32;">{message}</div>', unsafe_allow_html=True)


# data_analyst, sql_critic, sql_query_executor, insights_generator, insights_critic
if __name__ == "__main__":
    st.sidebar.title("Assistant Agents")
    st.sidebar.write("1. Planner")
    st.sidebar.write("2. SQL query generator")
    st.sidebar.write("3. SQL query critic")
    st.sidebar.write("4. SQL query executor")
    st.sidebar.write("5. Insights generator")
    st.sidebar.write("6. Insights critic")

    # st.sidebar.write("4. Image generator")
    # st.sidebar.write("4. Manager")

    st.markdown("""
    <div style='text-align: center; margin-top:-40px; margin-bottom: 5px;margin-left: -50px;'>
    <h2 style='font-size: 30px; font-family: Courier New, monospace;
                    letter-spacing: 2px; text-decoration: none;'>
    <img src="https://acis.affineanalytics.co.in/assets/images/logo_small.png" alt="logo" width="70" height="60">
    <span style='background: linear-gradient(45deg, #ed4965, #c05aaf);
                            -webkit-background-clip: text;
                            -webkit-text-fill-color: transparent;
                            text-shadow: none;'>
                    Affine - QUIN - AutoGen Multi-agent system
    </span>
    <span style='font-size: 40%;'>
    
    </span>
    </h2>
    </div>
    """, unsafe_allow_html=True)
    queue = Queue()

    def on_connect_wrapper(websocket):
        on_connect(websocket, queue)

    with IOWebsockets.run_server_in_thread(on_connect=on_connect_wrapper, port=8766) as uri:
        print(
            f" - test_setup() with websocket server running on {uri}.", flush=True)

        with ws_connect(uri) as websocket:
            print(f" - Connected to server on {uri}", flush=True)

            print(" - Sending message to server.", flush=True)
            # websocket.send("2+2=?")
            # time.sleep(1.5)

            user_query = st.text_input('Enter a query ')
            if st.button('Generate'):
                with st.spinner('Wait for it... Agents in action'):
                    ######################################################
                    # user_query_and_few_shot_example = []

                    few_shot_example_list = []
                    few_shot_example = similarity_search(user_query)
                    for example in few_shot_example:
                        few_shot_example_list.append(
                            (example["user_query"], example["sql_query"]))
                    user_query_and_few_shot_example = user_query + \
                        "###"+str(few_shot_example_list)
                    # user_query_and_few_shot_example.append(user_query)
                    # print("########################")
                    # print(few_shot_example_list)
                    # user_query_and_few_shot_example.append(
                    # str(few_shot_example_list))
                    # print("########################")
                    # print(user_query_and_few_shot_example)
                    ########################################################
                    # similar(user_query)
                    # sql_query != 'NA'
                    # pull few shot examples
                    # user_query = (query, few_shot)

                    websocket.send(user_query_and_few_shot_example)
                    while True:
                        # time.sleep(0.5)
                        message = websocket.recv()
                        message = message.decode(
                            "utf-8") if isinstance(message, bytes) else message

                        # print(message, end="", flush=True)
                        # st.write(message)
                        handle_message(message=message)

                        if "all-good-completed" in message.lower() or 'max-3-tries' in message.lower():
                            print()
                            print(
                                " - Received COMPLETED message. Exiting.", flush=True)
                            break

    if not queue.empty():
        # Define the directory name
        # dir_name = '.cache'

        # # Get the current working directory
        # current_dir = os.getcwd()

        # # Construct the full path to the directory
        # dir_path = os.path.join(current_dir, dir_name)

        # # Check if the directory exists
        # if os.path.exists(dir_path) and os.path.isdir(dir_path):
        #     # Remove the directory and its contents
        #     shutil.rmtree(dir_path)
        #     print(f"The directory '{dir_name}' has been deleted.")
        # else:
        #     print(f"The directory '{dir_name}' does not exist.")
        result_summary = queue.get()
        # st.write(f'result_summary:\n{result_summary}\n\n')
        output_ = summarize_chat_result(result_summary)
        manage_summary_format = [{"sql_query": "(generated_sql_query)", "insights": "(insights)", 'sql_critic_iter': '(Eg. 2)', 'sql_critic_messages': '(sql_critic_messages)',
                                  'insights_critic_iter': '(Eg. 1)', 'insights_critic_messages': '(insights_critic_messages)', 'no_of_tokens': '(no_of_tokens)', 'cost': '(cost)'}]
        output_ = output_[0]
        sql_query = output_['sql_query']
        insights = output_['insights']
        sql_critic_iter = output_['sql_critic_iter']
        sql_critic_messages = output_['sql_critic_messages']
        insights_critic_iter = output_['insights_critic_iter']
        insights_critic_messages = output_['insights_critic_messages']
        no_of_tokens = output_['no_of_tokens']
        cost = output_['cost']

        # uploading
        # spinner - updating our data model
        # successfully updated
        # st.write("all retrieved.")
        # Display SQL query as code
        SQL_query = sqlparse.format(
            sql_query, reindent=True, keyword_case='upper')
        # print("SQL_query: ", SQL_query)
        st.write('SQL Query:')
        st.code(SQL_query, language='sql')
        ################################################################
        few_shot_list = openai_chat(user_query, SQL_query, 3)
        few_shot_list.append(user_query)
        upload_to_search_index(few_shot_list, SQL_query,
                               database_name="quickinsight")
        ########################################################################
        # Display insights prominently
        st.markdown(
            f'<div style="font-size: 24px; font-weight: bold; color: #2e7d32;">Insights</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size: 18px; font-weight: bold; color: #2e7d32;">{insights}</div>', unsafe_allow_html=True)
        st.write("")
        st.write("SQL Critic Iteration:", sql_critic_iter)
        st.write("")
        st.write("SQL Critic Messages:", sql_critic_messages)
        st.write("")
        st.write("Insights Critic Iteration:", insights_critic_iter)
        st.write("")
        st.write("Insights Critic Messages:", insights_critic_messages)
        st.write("")
        st.write("Number of Tokens:", no_of_tokens)
        st.write("")
        st.write("Cost:", cost)
