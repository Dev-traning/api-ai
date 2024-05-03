import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from joblib import dump, load
from sklearn.pipeline import Pipeline
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
import os
from django.conf import settings
from rest_framework import generics
from rest_framework import status
from rest_framework.response import Response
from rest_framework.generics import ListCreateAPIView, RetrieveUpdateDestroyAPIView
from rest_framework.decorators import api_view
from rest_framework import status
from .models import TestData
from .Serializer import TestDataSerializer
from rest_framework.pagination import PageNumberPagination
from django.db.models import F

# Load the data from CSV
data_path  = os.path.join(settings.BASE_DIR, 'notebook', 'rd.csv')
data = pd.read_csv(data_path)
similarity_threshold = 0.4
greetings = ["hello", "hi", "hey", "Good morning", "good evening", "good after-noon", "namaste"]

# Preprocess the data
data['question'] = data['question'].apply(lambda x: x.lower())  # Convert to lowercase

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data['question'], data['answer'], test_size=0.2, random_state=42)

# Define a pipeline with both TfidfVectorizer and RandomForestClassifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', RandomForestClassifier())
])

# Train the model using the pipeline
pipeline.fit(X_train, y_train)

# Save the pipeline (including both the TfidfVectorizer and the model)
dump(pipeline, 'tfidf_vectorizer.joblib')

# Load the pipeline (including both the TfidfVectorizer and the model)
loaded_pipeline = load('tfidf_vectorizer.joblib')
# this method is used for predict ans using ai model 
def predict_answer(question):
    # Preprocess the question
    question = question.lower()
    entered_q = question
    # Check if the question is a greeting
    if question in greetings:
        return JsonResponse({
                "question": question,
                'answer': "Hello! I am chatbot of InfinityBrains. How can I assist you?"
            })

    try:
        # Predict the answer using the loaded pipeline
        answer = loaded_pipeline.predict([question])[0]

        # Calculate cosine similarity between input question and training questions
        question_tfidf = loaded_pipeline.named_steps['tfidf'].transform([question])
        similarities = cosine_similarity(question_tfidf, loaded_pipeline.named_steps['tfidf'].transform(X_train))

        # Find the maximum similarity and corresponding index
        max_similarity_index = similarities.argmax()
        max_similarity = similarities.max()

        # Retrieve the corresponding question and answer with the maximum similarity
        most_similar_question = X_train.iloc[max_similarity_index]
        most_similar_answer = y_train.iloc[max_similarity_index]

        # Check if the maximum similarity meets the threshold
        if max_similarity >= similarity_threshold:
            return JsonResponse({
                "question": entered_q,
                "answer": most_similar_answer,
                "max_similarity": max_similarity,
                'most_similar_question': most_similar_question,
                "most_similar_answer": most_similar_answer
            })
        else:
            return JsonResponse({
                "question": entered_q,
                "answer": "As of 2024, I'm not familiar enough with that topic to provide an accurate response. Is there anything else I can help you with?"
            })
    except Exception as e:
        return JsonResponse({
            "error": str(e)  # Return the error message
        })

# Define the path to the trained model file
model_file_path = os.path.join(settings.BASE_DIR, 'notebook', 'tfidf_vectorizer.joblib')

# Load the pipeline (including both the TfidfVectorizer and the model)
loaded_pipeline = load(model_file_path)

# Define greetings
greetings = ["hello", "hi", "hey", "good morning", "good evening", "good after-noon", "namaste"]


@csrf_exempt
# used for ask question using api endpint 
def ask_question(request):
    if request.method == 'POST':
        # Get the question from the request data
        question = request.POST.get('question', '')

        # Check if the question is empty
        if not question:
            return HttpResponseBadRequest('Question is required.')

        # Preprocess the question
        question = question.lower()

        # Check if the question is a greeting
        if question in greetings:
            return JsonResponse({
                "question":question,
                'answer': "Hello! I am chatbot of InfinityBrains. How can I assist you?"
                })

        # Predict the answer using the loaded pipeline
        answer = predict_answer(question)

        # Return the answer in the response
        return answer  
    else:
        # Return an error response for unsupported request method
        return HttpResponseBadRequest('Only POST method is allowed for this endpoint.')

@api_view(['GET', 'POST'])
# add data in databse
def testdata_list_create(request):
    paginator = CustomPagination()
    queryset = TestData.objects.all()
    paginated_queryset = paginator.paginate_queryset(queryset, request)
    if request.method == 'GET':
       # Filtering
        user_id = request.query_params.get('user_id')
        queryset = TestData.objects.all()
        if user_id:
            queryset = queryset.filter(user_id=user_id)
        
         # Pagination
        paginator = CustomPagination()
        paginated_queryset = paginator.paginate_queryset(queryset, request)
        
        serializer = TestDataSerializer(paginated_queryset, many=True)
        response = paginator.get_paginated_response(serializer.data)
        # Check if results are empty
        if not response.data['results']:
            return Response({
                'status':False,
                "status_code":204,
                'message': 'No data available'}, 
                status=status.HTTP_204_NO_CONTENT)
        
        return response

    elif request.method == 'POST':
        serializer = TestDataSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return custom_error_response(serializer.errors)

@api_view(['GET', 'PUT', 'DELETE'])
# get and update delete data with id
def testdata_retrieve_update_destroy(request, pk):
    try:
        testdata = TestData.objects.get(pk=pk)
    except TestData.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    if request.method == 'GET':
        serializer = TestDataSerializer(testdata)
        return Response(serializer.data)

    elif request.method in ['PUT', 'PATCH']:
        serializer = TestDataSerializer(testdata, data=request.data, partial=True)  # Specify partial=True
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == 'DELETE':
        testdata.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

# error Response custom for from 
def custom_error_response(error_data):
    # Extracting only field names with errors
    error_fields = list(error_data.keys())
    
    # Construct response JSON
    response_data = {
    "status": False,
    "status_code": status.HTTP_400_BAD_REQUEST,
    "message": "This field's required: " + ", ".join(error_fields)
}


    # Return response
    return Response(response_data, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def store_data(request):
    serializer = TestDataSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    else:
        return custom_error_response(serializer.errors)

class CustomPagination(PageNumberPagination):
    page_size =  9
    page_size_query_param = 'per_page'