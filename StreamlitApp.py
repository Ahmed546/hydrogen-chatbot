from src.helper import load_db, load_pretrained_model, chat_bot, chain_type_kwargs
import streamlit as st


db = load_db()
llm = load_pretrained_model()
qa = chat_bot(db, llm, chain_type_kwargs)


user_input = st.text_input("Your question:")


if st.button("Get Answer"):
    if user_input:
        result = qa.invoke({"query": user_input})
        st.write("Response:", result["result"])
    else:
        st.warning("Please enter a question.")


if 'history' not in st.session_state:
    st.session_state.history = []


if user_input:
    st.session_state.history.append(f"You: {user_input}")
if 'result' in locals():
    st.session_state.history.append(f"Chatbot: {result['result']}")


if st.session_state.history:
    st.write("### Chat History")
    for message in st.session_state.history:
        st.write(message)

if __name__ == '__main__':
    
    pass 
