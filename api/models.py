# models.py

from django.db import models

class TestData(models.Model):
    id = models.AutoField(primary_key=True)
    user_id = models.IntegerField()
    question = models.TextField()
    answer = models.TextField()
    max_similarity = models.FloatField()
    most_similar_question = models.TextField()
    most_similar_answer = models.TextField()

    def __str__(self):
        return self.question  
