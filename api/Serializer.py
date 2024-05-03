from rest_framework import serializers
from rest_framework.response import Response
from rest_framework import status
from .models import TestData

class TestDataSerializer(serializers.ModelSerializer):
    def __init__(self, *args, **kwargs):
        super(TestDataSerializer, self).__init__(*args, **kwargs)
        self.fields['user_id'].error_messages['required'] = 'User ID is required.'
        self.fields['question'].error_messages['required'] = 'Question is required.'
        self.fields['answer'].error_messages['required'] = 'Answer is required.'
        self.fields['max_similarity'].error_messages['required'] = 'Max similarity is required.'
        self.fields['most_similar_question'].error_messages['required'] = 'Most similar question is required.'
        self.fields['most_similar_answer'].error_messages['required'] = 'Most similar answer is required.'

    class Meta:
        model = TestData
        fields = "__all__" 