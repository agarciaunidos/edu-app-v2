# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger
from llm_bedrock import retrieval_answer

LOGGER = get_logger(__name__)


def run():
  st.title("Education AI Program")
  st.caption("A Digital Services - Education Project")

  if "messages" not in st.session_state:
      st.session_state["messages"] = [{"role": "assistant", "content": "Ask your query..."}]

  for msg in st.session_state.messages:
      st.chat_message(msg["role"]).write(msg["content"])


  with st.sidebar:
      llm_model = st.selectbox("Select LLM", ["Anthropic Claude V2","GPT-4-1106-preview"])
      vector_store = st.selectbox("Select Vector DB", ["Pinecone: Highschool democracy", "Pinecone: University of Arizona"])

  if prompt := st.chat_input():
      if len(prompt) > 0:
          st.info("Your Query: " + prompt)
          answer = retrieval_answer(prompt, llm_model,vector_store)
          st.success(answer)
      else:
          st.error("Please enter a query.")


if __name__ == "__main__":
    run()
