def analyze_image(file, file_info) :
    # TODO: 현재 테스트용. 실제 모델 추론 로직을 여기에 넣을 예정
    result = {
        "file_description": "purple and blue sunset over a body of water",
        "file_moods": [
            {"label": "몽환적", "score": 21.78},
            {"label": "설렘", "score": 11.14},
            {"label": "후련함", "score": 10.06}
        ]
    }
    return result
