from django.urls import path
from .views import ask_question, testdata_list_create, testdata_retrieve_update_destroy, store_data

urlpatterns = [
    path('ask/', ask_question, name='ask_question'),
    path('testdata/', testdata_list_create, name='testdata-list-create'),  # Fixed mapping
    path('testdata/<int:pk>/', testdata_retrieve_update_destroy, name='testdata-retrieve-update-destroy'),
    path('store-data/', store_data, name='store-data'),
]
