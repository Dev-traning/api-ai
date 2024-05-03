from django.contrib import admin
from .models import TestData

class TestDataAdmin(admin.ModelAdmin):
    # Define a function to return all fields of the model
    def get_list_display(self, request):
        return [field.name for field in TestData._meta.get_fields()]

# Register your model with the custom admin class
admin.site.register(TestData, TestDataAdmin)
