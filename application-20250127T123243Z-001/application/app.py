import streamlit as st
import joblib
import numpy as np    
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import base64
from streamlit.components.v1 import html

# Load text models
tfidf_model = joblib.load(r'E:\DEPI\FINAL PROJECT\models\text_models\tfidf_vectorizer.pkl')
log_model = joblib.load(r'E:\DEPI\FINAL PROJECT\models\text_models\Log_model.pkl')
# random_forest_model = joblib.load(r'E:\DEPI\FINAL PROJECT\models\text_models\random_forest_model.pkl')


# Load features model
scaler_model = joblib.load(r'E:\DEPI\FINAL PROJECT\models\features_models\scaler.pkl')
logisticRegression = joblib.load(r'E:\DEPI\FINAL PROJECT\models\features_models\logisticRegression_model.pkl')
# AdaBoost_model = joblib.load(r'E:\DEPI\FINAL PROJECT\models\features_models\Ada_boost.pkl')
# XG_boost_model = joblib.laod(r'E:\DEPI\FINAL PROJECT\models\features_models\xgboost_model.pkl')



home_page_background_path = r'E:\DEPI\FINAL PROJECT\Application\pictures\home_page_image.jpg'
selection_page_background_path = r'E:\DEPI\FINAL PROJECT\Application\pictures\selection_page_image.png'


# Function to apply custom CSS for background image using base64
def apply_background_image(image_path):
    # Load the image and encode it to base64
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode()

    # Apply the CSS with the base64 image
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{base64_image}");
            background-size: 100% 100%; /* Adjust the size here (width height) */
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """, unsafe_allow_html=True)



# Main Home page for model selection
def home_page():
    # Apply background image
    apply_background_image(home_page_background_path)

    # Apply custom CSS to center and style the title
    st.markdown("""
        <style>
        .centered-title {
            color: black;
            font-size: 40px;
            font-weight: bold;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)

    # Render the title using custom class
    st.markdown('<h2 class="centered-title">Fraud Detection in Financial Transaction</h2>', unsafe_allow_html=True)
    
    # Add space to push the button down
    for _ in range(15):
        st.write("")  # Add empty space
    
    # Create columns to center the button
    col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 1, 1, 1, 1])  # Adjust the ratios to control the width

    with col4:  # Put the button in the center column
        if st.button("Start"):
            st.session_state['page'] = 'type_selection'



def type_selection():
    apply_background_image(selection_page_background_path)  
    
    # Dropdown to Select The Transaction Type
    option = st.selectbox(
        "Select The Transaction Type:",
        ("None", "Predict With Text Mail", "Predict With Transaction Credit Card")
    )
            
    if option == 'None':
        st.write('Please Choose!')
        
    # Button to navigate to the selected page
    if st.button("Go to Prediction"): 
        # Disable the button if no transaction type is selected
        if option == 'None':
            st.error("Please select a transaction type to proceed.")
        else:
            # Store selected option in session state to navigate to the corresponding page
            if option == "Predict With Text Mail":
                st.session_state["page"] = "text_mail"
            elif option == "Predict With Transaction Credit Card":
                st.session_state["page"] = "credit_card"
            
            st.experimental_rerun()  # Refresh the app to switch the page based on the selected option
    if st.button('Back To Home'):
            st.session_state["page"] = 'home'
    
    

def predict_with_text_mail():  # page
    apply_background_image(selection_page_background_path)  
    
    st.subheader("Predict With Text Mail")
    
    # Text input for user to paste email content
    email_content = st.text_area("Enter the email content for fraud prediction:")
    
    if st.button("Predict"):
        # Validate email content and model selection
        if not email_content.strip():
            st.error("Please enter email content.")
        else:
            prediction = None  # Initialize prediction variable
            
            # If inputs are valid, proceed with the prediction
            transformed_text = tfidf_model.transform([email_content])
            prediction = log_model.predict(transformed_text)
            
            # Display the result if prediction exists
            if prediction is not None:
                if prediction[0] == 1:
                    st.success("This transaction is likely to be fraudulent.")
                else:
                    st.success("This transaction seems legitimate.")

    if st.button('Back'):
        st.session_state["page"] = 'back'




# Preload label encoders for categorical features (use the same encoder for consistency)
use_chip_encoder = LabelEncoder()
merchant_city_encoder = LabelEncoder()
merchant_state_encoder = LabelEncoder()
notes_encoder = LabelEncoder()

# Fit the label encoders with known data categories
use_chip_encoder.fit(['Swipe Transaction', 'Chip Transaction'])
merchant_city_encoder.fit(['La Verne', 'Mira Loma', 'Carrizo Springs', 'Jenkintown', 'King Of Prussia',
                           'Star City', 'Las Vegas', 'Miami'])  # Include all unique city names here
merchant_state_encoder.fit(['CA', 'TX', 'NJ', 'UT', 'FL', 'NV', 'HI', 'NY', 'MA', 'MI', 'MN', 'IA',
                            'IL', 'WA', 'SC', 'AK', 'CO', 'NC', 'OH', 'ME', 'MO', 'PA', 'AR',
                            'GA', 'CT', 'AZ', 'DC', 'KY', 'AL', 'OR', 'WI', 'IN', 'MD', 'VA',
                            'TN', 'LA', 'RI', 'OK', 'KS', 'ND', 'NM', 'MS', 'NH', 'NE', 'WV',
                            'ID', 'DE', 'WY', 'SD', 'MT', 'VT'])  # Include all states
notes_encoder.fit(['Technical Glitch', 'Insufficient Balance', 'Bad PIN',
                   'Bad PIN,Insufficient Balance', 'Bad PIN,Technical Glitch',
                   'Bad Zipcode', 'Insufficient Balance,Technical Glitch',
                   'Bad Zipcode,Insufficient Balance', 'Bad Zipcode,Technical Glitch',
                   'Bad CVV', 'Bad Expiration', 'Bad Card Number',
                   'Bad Card Number,Insufficient Balance'])


# Function to handle predictions using credit card transaction data  
def predict_with_transaction_credit_card():  # page
    apply_background_image(selection_page_background_path)  
    
    st.subheader("Predict With Transaction Credit Card")
    
    # Inputs for the transaction features
    Year = st.number_input('Year', min_value=2000, max_value=2024, value=2023)
    Month = st.number_input('Month', min_value=1, max_value=12, value=1)
    Day = st.number_input('Day', min_value=1, max_value=31, value=1)
    Hours = st.number_input('Hours', min_value=0.0, max_value=24.0, value=0.0)
    Amount = st.number_input('Amount', min_value=0.0, max_value=1000000.0, value=0.0) 
    Zip = st.number_input('Zip', min_value=0, max_value=99999, value=10000)
    MCC = st.number_input('MCC', min_value=1000, max_value=9999, value=4000)
    
    transaction_type = st.selectbox("Select the transaction type:",
                                    ("Swipe Transaction", "Chip Transaction"))
    Merchant_City = st.selectbox("Select the Merchant City:",
                                ('La Verne', 'Mira Loma', 'Carrizo Springs', 'Jenkintown', 'King Of Prussia',
                                 'Star City', 'Las Vegas', 'Miami'))
    Merchant_State = st.selectbox("Select the Merchant State:",
                                  ('CA', 'TX', 'NJ', 'UT', 'FL', 'NV', 'HI', 'NY', 'MA', 'MI', 'MN', 'IA',
                                   'IL', 'WA', 'SC', 'AK', 'CO', 'NC', 'OH', 'ME', 'MO', 'PA', 'AR', 
                                   'GA', 'CT', 'AZ', 'DC', 'KY', 'AL', 'OR', 'WI', 'IN', 'MD', 'VA', 
                                   'TN', 'LA', 'RI', 'OK', 'KS', 'ND', 'NM', 'MS', 'NH', 'NE', 'WV',
                                   'ID', 'DE', 'WY', 'SD', 'MT', 'VT'))
    Notes_Error = st.selectbox("Select the Notes Error:",
                               ('Technical Glitch', 'Insufficient Balance', 'Bad PIN', 
                                'Bad PIN,Insufficient Balance', 'Bad PIN,Technical Glitch', 
                                'Bad Zipcode', 'Insufficient Balance,Technical Glitch', 
                                'Bad Zipcode,Insufficient Balance', 'Bad Zipcode,Technical Glitch',
                                'Bad CVV', 'Bad Expiration', 'Bad Card Number', 
                                'Bad Card Number,Insufficient Balance'))

    # When the 'Predict' button is pressed
    if st.button("Predict"):
        # Create a new data point dictionary to hold transaction features
        new_data_point = {
            'Year': Year,
            'Month': Month,
            'Day': Day,
            'Hours': Hours,
            'Amount': Amount,
            'Use_Chip': transaction_type,
            'Merchant_City': Merchant_City,
            'Merchant_State': Merchant_State,
            'Zip': Zip,
            'MCC': MCC,
            'Notes': Notes_Error
        }

        # Load the scaler and model from joblib 
        scaler = scaler_model
        model = logisticRegression

        # Use the predict_fraud function to predict fraud based on the new transaction data
        prediction = predict_fraud(model, new_data_point, scaler)

        # Display the prediction result
        if prediction == 1:
            st.success("This transaction is likely to be fraudulent.")
        else:
            st.success("This transaction seems legitimate.")

    if st.button('Back'):
        st.session_state["page"] = 'back'

        
        
# Prediction function
def predict_fraud(model, new_data_point, scaler):
    
    # Convert categorical text features to numerical values
    new_data_point['Use_Chip'] = use_chip_encoder.transform([new_data_point['Use_Chip']])[0]
    new_data_point['Merchant_City'] = merchant_city_encoder.transform([new_data_point['Merchant_City']])[0]
    new_data_point['Merchant_State'] = merchant_state_encoder.transform([new_data_point['Merchant_State']])[0]
    new_data_point['Notes'] = notes_encoder.transform([new_data_point['Notes']])[0]

    # Prepare the input features in the correct order
    features = [
        new_data_point['Year'],
        new_data_point['Month'],
        new_data_point['Day'],
        new_data_point['Hours'],
        new_data_point['Amount'],
        new_data_point['Use_Chip'],
        new_data_point['Merchant_City'],
        new_data_point['Merchant_State'],
        new_data_point['Zip'],
        new_data_point['MCC'],
        new_data_point['Notes']
    ]

    # Ensure the input is a numpy array and has the correct shape
    features = np.array(features).reshape(1, -1)

    # Apply scaling 
    features = scaler.transform(features)

    # Make a prediction using the model
    prediction = model.predict(features)[0]

    return prediction


# Main function to control page navigation
def main():
    # Initialize session state if not already present
    if "page" not in st.session_state:
        st.session_state["page"] = "home"
    
    # Navigate between pages based on session state
    if st.session_state["page"] == "home":
        home_page()
    elif st.session_state["page"] == "type_selection" or st.session_state["page"] == "back":
        type_selection()
    elif st.session_state["page"] == "text_mail":
        predict_with_text_mail()
    elif st.session_state["page"] == "credit_card":
        predict_with_transaction_credit_card()


# Run the app
if __name__ == "__main__":
    main()
