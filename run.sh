#!/bin/bash

# 프로젝트 디렉터리로 이동
cd Projects/so_dam_server

# 가상 환경 활성화
source venv/bin/activate

# Django 개발 서버 실행
echo "Starting Django development server..."
python manage.py runserver