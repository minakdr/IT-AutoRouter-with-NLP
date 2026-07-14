import streamlit as st
import joblib

# 1. Set up the page title and description
st.title("IT Support Ticket Auto-Router")
st.write("Type a sample IT ticket below, and our Machine Learning AI will automatically route it to the correct department!")

# 2. Load the trained pipeline (The Translator + The Brain)
# The @st.cache_resource tells the website to only load the 5,000-word dictionary ONCE, 
# keeping the app lightning-fast every time you click the button.
@st.cache_resource
def load_model():
    # Make sure this matches the exact name of the file you saved earlier!
    return joblib.load('ticket_routing_pipeline.pkl')

# Try to load the model, but show a friendly error if the file is missing
try:
    pipeline = load_model()
except FileNotFoundError:
    st.error("🚨 Error: Could not find 'ticket_routing_pipeline.pkl'. Make sure it is in the same folder as this app.py file!")
    st.stop()

# 3. Create a text box for the user to type into
user_ticket = st.text_area(
    "Enter your IT issue here:", 
    placeholder="e.g., I need a new mouse for my laptop, mine is broken."
)

# 4. Create the "Submit" button
if st.button("Route Ticket"):
    if user_ticket.strip() == "":
        st.warning("⚠️ Please type a ticket description first.")
    else:
        # 5. Make the prediction!
        # We put user_ticket inside brackets [ ] because the pipeline expects a list of tickets.
        # We put [0] at the end to pull the text string out of the prediction list.
        prediction = pipeline.predict([user_ticket])[0]
        
        # 6. Display the final answer on the screen beautifully
        st.success(f"✅ Automatically Sent to: **{prediction}**")
        
