#!/bin/bash

# 가상 환경 활성화
echo "--- Activating virtual environment ---"
source venv/bin/activate

# Django 개발 서버 실행
echo "--- Starting Django development server ---"
# python manage.py runserver
python manage.py runserver 0.0.0.0:8000

# 스크립트 종료 후 가상 환경 비활성화
deactivate