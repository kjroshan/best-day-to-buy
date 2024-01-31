import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import TextOperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

def read_coupon_file(file_path):
    try:
        with open(file_path, 'r') as file:
            coupons = [line.strip() for line in file.readlines()]
        return coupons
    except FileNotFoundError:
        print(f"Error: Coupon file '{file_path}' not found.")
        return []

def extract_text_from_image(image_path, cv_client):
    with open(image_path, "rb") as image_stream:
        result = cv_client.recognize_printed_text_in_stream(image_stream)
        operation_id = result.headers["Operation-Location"].split("/")[-1]

        # Wait for the operation to complete
        while True:
            status = cv_client.get_text_operation_result(operation_id)
            if status.status not in [TextOperationStatusCodes.SUCCEEDED, TextOperationStatusCodes.FAILED]:
                time.sleep(1)
            else:
                break

        if status.status == TextOperationStatusCodes.SUCCEEDED:
            return [region.text for region in status.recognition_results[0].lines]
        else:
            return []

def validate_coupon_with_azure(coupon_text, text_analytics_client):
    response = text_analytics_client.analyze_sentiment(coupon_text)
    sentiment = response[0].sentiment

    # You can customize this logic based on your specific requirements
    if sentiment == 'positive':
        return True
    else:
        return False

def train_price_prediction_model(df):
    # Feature engineering (you may need to include more relevant features)
    X = df[['DayOfYear']]
    y = df['Price']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    return model

def find_day_of_lowest_price(df, model):
    # Create a dataframe with all days of the year
    all_days = pd.DataFrame({'DayOfYear': range(1, 366)})

    # Predict prices for all days
    predicted_prices = model.predict(all_days[['DayOfYear']])

    # Find the day with the lowest predicted price
    day_of_lowest_price = all_days.loc[predicted_prices.argmin()]

    return day_of_lowest_price['DayOfYear'].item()

def main():
    # Replace 'YOUR_COMPUTER_VISION_KEY', 'YOUR_COMPUTER_VISION_ENDPOINT', 'YOUR_TEXT_ANALYTICS_KEY', and 'YOUR_TEXT_ANALYTICS_ENDPOINT' with your actual Azure keys and endpoints
    cv_key = 'YOUR_COMPUTER_VISION_KEY'
    cv_endpoint = 'YOUR_COMPUTER_VISION_ENDPOINT'
    text_analytics_key = 'YOUR_TEXT_ANALYTICS_KEY'
    text_analytics_endpoint = 'YOUR_TEXT_ANALYTICS_ENDPOINT'

    # Replace 'path/to/your/image.jpg' with the actual path to your scanned coupon image
    image_path = 'path/to/your/image.jpg'

    # Read coupon file
    coupon_file_path = "coupons.txt"  # Change this to the path of your coupon file
    valid_coupons = read_coupon_file(coupon_file_path)

    if not valid_coupons:
        print("No valid coupons found. Exiting.")
        return

    # Set up Azure clients
    cv_client = ComputerVisionClient(cv_endpoint, CognitiveServicesCredentials(cv_key))
    text_analytics_client = TextAnalyticsClient(endpoint=text_analytics_endpoint, credential=AzureKeyCredential(text_analytics_key))

    # Extract text from the scanned image
    coupon_text = extract_text_from_image(image_path, cv_client)

    if not coupon_text:
        print("Unable to extract text from the image. Exiting.")
        return

    print("Extracted Text from Image:")
    print("\n".join(coupon_text))

    user_coupon = " ".join(coupon_text).strip()

    # Validate coupon with Azure Text Analytics
    if validate_coupon_with_azure(user_coupon, text_analytics_client):
        print("Congratulations! Your coupon is valid.")

        # Load historical pricing data (replace this with your actual historical pricing data)
        data = {
            'Date': pd.date_range(start='2022-01-01', end='2023-12-31', freq='D'),
            'Price': np.random.randint(50, 200, size=(730,)),
        }

        df = pd.DataFrame(data)
        df['DayOfYear'] = df['Date'].dt.dayofyear

        # Train a linear regression model for price prediction
        price_prediction_model = train_price_prediction_model(df)

        # Find the day of the year with the lowest predicted price
        day_of_lowest_price = find_day_of_lowest_price(df, price_prediction_model)
        lowest_price_date = datetime.strptime(str(day_of_lowest_price), "%j")

        print(f"Predicted Day of Lowest Price: {lowest_price_date.strftime('%Y-%m-%d')}")

        # Add your discount logic or further actions here based on the coupon and price prediction
    else:
        print("Sorry, the entered coupon code is not valid.")

if __name__ == "__main__":
    main()
