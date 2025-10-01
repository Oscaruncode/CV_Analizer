import requests

url = "http://localhost:8000/SentimentAnalize"

payload = {
  "questions": {
    "1": "¿Cómo percibes el ambiente laboral?",
    "2": "¿Recibes apoyo de tu equipo?",
    "3": "¿Qué opinas de tu carga laboral?",
    "4": "¿Cómo describirías la comunicación con tus jefes?",
    "5": "¿Qué piensas de las oportunidades de crecimiento?"
  },
    "responses": [
        {"AnswerID": 1, "QuestionId": 1, "AnswerText": "El ambiente es excelente, me siento motivado"},
        {"AnswerID": 2, "QuestionId": 1, "AnswerText": "El ambiente es malo, mucha tensión entre compañeros"},
        {"AnswerID": 3, "QuestionId": 1, "AnswerText": "El ambiente es regular, a veces bien, a veces mal"},
        {"AnswerID": 4, "QuestionId": 2, "AnswerText": "Mi equipo siempre me apoya y colabora"},
        {"AnswerID": 5, "QuestionId": 2, "AnswerText": "Nunca recibo ayuda, cada uno se preocupa solo por sí mismo"},
        {"AnswerID": 6, "QuestionId": 2, "AnswerText": "A veces recibo apoyo, otras no, depende del día"},
        {"AnswerID": 7, "QuestionId": 3, "AnswerText": "La carga laboral es excesiva e inhumana"},
        {"AnswerID": 8, "QuestionId": 3, "AnswerText": "Tengo una carga laboral adecuada y balanceada"},
        {"AnswerID": 9, "QuestionId": 3, "AnswerText": "La carga es alta pero manejable"},
        {"AnswerID": 10, "QuestionId": 4, "AnswerText": "Mis jefes son claros y escuchan mis inquietudes"},
        {"AnswerID": 11, "QuestionId": 4, "AnswerText": "Mis jefes nunca escuchan y la comunicación es pésima"},
        {"AnswerID": 12, "QuestionId": 4, "AnswerText": "La comunicación es aceptable, aunque podría mejorar"},
        {"AnswerID": 13, "QuestionId": 5, "AnswerText": "Tengo oportunidades de ascenso y capacitación"},
        {"AnswerID": 14, "QuestionId": 5, "AnswerText": "Aquí no hay futuro, todo está estancado"},
        {"AnswerID": 15, "QuestionId": 5, "AnswerText": "No lo sé, no tengo mucha información"},
        {"AnswerID": 16, "QuestionId": 1, "AnswerText": "El ambiente es cálido y de confianza"},
        {"AnswerID": 17, "QuestionId": 1, "AnswerText": "El ambiente es muy tóxico"},
        {"AnswerID": 18, "QuestionId": 1, "AnswerText": "Es un ambiente indiferente, nada especial"},
        {"AnswerID": 19, "QuestionId": 2, "AnswerText": "Siempre que necesito algo, mi equipo me respalda"},
        {"AnswerID": 20, "QuestionId": 2, "AnswerText": "Mi equipo nunca está disponible"},
        {"AnswerID": 21, "QuestionId": 2, "AnswerText": "Algunas veces sí, otras veces no"},
        {"AnswerID": 22, "QuestionId": 3,
         "AnswerText": "La carga es justa y me permite conciliar con mi vida personal"},
        {"AnswerID": 23, "QuestionId": 3, "AnswerText": "Es demasiado trabajo, me siento agotado"},
        {"AnswerID": 24, "QuestionId": 3, "AnswerText": "La carga varía mucho"},
        {"AnswerID": 25, "QuestionId": 4, "AnswerText": "Los jefes comunican todo con claridad"},
        {"AnswerID": 26, "QuestionId": 4, "AnswerText": "Los jefes nunca explican nada"},
        {"AnswerID": 27, "QuestionId": 4, "AnswerText": "La comunicación depende de la situación"},
        {"AnswerID": 28, "QuestionId": 5, "AnswerText": "Siento que puedo crecer y aprender"},
        {"AnswerID": 29, "QuestionId": 5, "AnswerText": "Aquí nunca hay posibilidades de crecimiento"},
        {"AnswerID": 30, "QuestionId": 5, "AnswerText": "No estoy seguro"},

        {"AnswerID": 31, "QuestionId": 1, "AnswerText": "El ambiente laboral fomenta la colaboración"},
        {"AnswerID": 32, "QuestionId": 1, "AnswerText": "El ambiente es competitivo y estresante"},
        {"AnswerID": 33, "QuestionId": 2, "AnswerText": "Mi equipo comparte conocimientos y me ayuda a mejorar"},
        {"AnswerID": 34, "QuestionId": 2, "AnswerText": "El apoyo del equipo es inconsistente"},
        {"AnswerID": 35, "QuestionId": 3, "AnswerText": "Mi carga laboral es manejable con planificación"},
        {"AnswerID": 36, "QuestionId": 3, "AnswerText": "A veces hay demasiada presión y plazos difíciles"},
        {"AnswerID": 37, "QuestionId": 4, "AnswerText": "La comunicación es abierta y transparente"},
        {"AnswerID": 38, "QuestionId": 4, "AnswerText": "Los jefes comunican solo lo estrictamente necesario"},
        {"AnswerID": 39, "QuestionId": 5, "AnswerText": "Existen programas de desarrollo profesional"},
        {"AnswerID": 40, "QuestionId": 5, "AnswerText": "Siento que mi crecimiento está limitado"},

        {"AnswerID": 41, "QuestionId": 1, "AnswerText": "El ambiente laboral me inspira a dar lo mejor"},
        {"AnswerID": 42, "QuestionId": 1, "AnswerText": "El ambiente laboral es monótono y aburrido"},
        {"AnswerID": 43, "QuestionId": 2, "AnswerText": "Siempre encuentro apoyo cuando lo necesito"},
        {"AnswerID": 44, "QuestionId": 2, "AnswerText": "El equipo rara vez colabora"},
        {"AnswerID": 45, "QuestionId": 3, "AnswerText": "Mi carga laboral es demasiado ligera"},
        {"AnswerID": 46, "QuestionId": 3, "AnswerText": "La carga es impredecible y causa estrés"},
        {"AnswerID": 47, "QuestionId": 4, "AnswerText": "Los jefes fomentan feedback constante"},
        {"AnswerID": 48, "QuestionId": 4, "AnswerText": "La comunicación es confusa y poco clara"},
        {"AnswerID": 49, "QuestionId": 5, "AnswerText": "Puedo avanzar si me esfuerzo"},
        {"AnswerID": 50, "QuestionId": 5, "AnswerText": "No hay programas de desarrollo, todo depende del azar"},

        {"AnswerID": 51, "QuestionId": 1, "AnswerText": "El ambiente es creativo y estimulante"},
        {"AnswerID": 52, "QuestionId": 2, "AnswerText": "Mi equipo es colaborativo pero a veces lento"},
        {"AnswerID": 53, "QuestionId": 3, "AnswerText": "La carga laboral fluctúa según la temporada"},
        {"AnswerID": 54, "QuestionId": 4, "AnswerText": "Algunos jefes comunican muy bien, otros no"},
        {"AnswerID": 55, "QuestionId": 5, "AnswerText": "El crecimiento depende de la iniciativa individual"},

        {"AnswerID": 56, "QuestionId": 1, "AnswerText": "El ambiente laboral es estimulante y creativo"},
        {"AnswerID": 57, "QuestionId": 1, "AnswerText": "El ambiente es competitivo pero motivador"},
        {"AnswerID": 58, "QuestionId": 1, "AnswerText": "El ambiente laboral es monótono"},
        {"AnswerID": 59, "QuestionId": 1, "AnswerText": "Siento que hay tensión entre compañeros"},
        {"AnswerID": 60, "QuestionId": 1, "AnswerText": "El ambiente es colaborativo y positivo"},

        {"AnswerID": 61, "QuestionId": 2, "AnswerText": "Mi equipo siempre me respalda"},
        {"AnswerID": 62, "QuestionId": 2, "AnswerText": "A veces recibo ayuda, otras veces no"},
        {"AnswerID": 63, "QuestionId": 2, "AnswerText": "Nunca encuentro apoyo de mi equipo"},
        {"AnswerID": 64, "QuestionId": 2, "AnswerText": "El equipo comparte conocimientos de forma regular"},
        {"AnswerID": 65, "QuestionId": 2, "AnswerText": "Recibo apoyo solo cuando lo pido explícitamente"},

        {"AnswerID": 66, "QuestionId": 3, "AnswerText": "La carga laboral es manejable y equilibrada"},
        {"AnswerID": 67, "QuestionId": 3, "AnswerText": "Siento demasiada presión y estrés"},
        {"AnswerID": 68, "QuestionId": 3, "AnswerText": "Mi carga varía según el día"},
        {"AnswerID": 69, "QuestionId": 3, "AnswerText": "Tengo trabajo suficiente, ni demasiado ni poco"},
        {"AnswerID": 70, "QuestionId": 3, "AnswerText": "La carga es excesiva y difícil de manejar"},

        {"AnswerID": 71, "QuestionId": 4, "AnswerText": "Los jefes comunican todo claramente"},
        {"AnswerID": 72, "QuestionId": 4, "AnswerText": "La comunicación con los jefes es confusa"},
        {"AnswerID": 73, "QuestionId": 4, "AnswerText": "Algunos jefes explican bien, otros no"},
        {"AnswerID": 74, "QuestionId": 4, "AnswerText": "Los jefes escuchan mis ideas y sugerencias"},
        {"AnswerID": 75, "QuestionId": 4, "AnswerText": "Nunca recibo feedback de mis superiores"},

        {"AnswerID": 76, "QuestionId": 5, "AnswerText": "Existen muchas oportunidades de crecimiento"},
        {"AnswerID": 77, "QuestionId": 5, "AnswerText": "No veo posibilidades de desarrollo profesional"},
        {"AnswerID": 78, "QuestionId": 5, "AnswerText": "Estoy inseguro sobre las oportunidades de crecimiento"},
        {"AnswerID": 79, "QuestionId": 5, "AnswerText": "Siento que puedo mejorar mis habilidades y avanzar"},
        {"AnswerID": 80, "QuestionId": 5, "AnswerText": "El crecimiento depende de la iniciativa personal"},

        {"AnswerID": 81, "QuestionId": 1, "AnswerText": "El ambiente fomenta la creatividad y colaboración"},
        {"AnswerID": 82, "QuestionId": 1, "AnswerText": "El ambiente es indiferente y poco motivador"},
        {"AnswerID": 83, "QuestionId": 1, "AnswerText": "Siento que hay rivalidad entre compañeros"},
        {"AnswerID": 84, "QuestionId": 1, "AnswerText": "El ambiente laboral me anima a superarme"},
        {"AnswerID": 85, "QuestionId": 1, "AnswerText": "Hay momentos de tensión pero también cooperación"},

        {"AnswerID": 86, "QuestionId": 2, "AnswerText": "Mi equipo colabora de manera constante"},
        {"AnswerID": 87, "QuestionId": 2, "AnswerText": "Recibo ayuda solo cuando es necesario"},
        {"AnswerID": 88, "QuestionId": 2, "AnswerText": "El equipo rara vez coopera"},
        {"AnswerID": 89, "QuestionId": 2, "AnswerText": "A veces hay apoyo, otras veces no"},
        {"AnswerID": 90, "QuestionId": 2, "AnswerText": "El equipo siempre está dispuesto a ayudar"},

        {"AnswerID": 91, "QuestionId": 3, "AnswerText": "La carga laboral es adecuada para mi rendimiento"},
        {"AnswerID": 92, "QuestionId": 3, "AnswerText": "Siento que tengo demasiado trabajo y poco tiempo"},
        {"AnswerID": 93, "QuestionId": 3, "AnswerText": "Mi carga laboral varía mucho según la temporada"},
        {"AnswerID": 94, "QuestionId": 3, "AnswerText": "Tengo trabajo ligero y puedo enfocarme en calidad"},
        {"AnswerID": 95, "QuestionId": 3, "AnswerText": "A veces la carga es pesada, otras ligera"},

        {"AnswerID": 96, "QuestionId": 4, "AnswerText": "La comunicación con los jefes es abierta y transparente"},
        {"AnswerID": 97, "QuestionId": 4, "AnswerText": "No recibo suficiente información de mis superiores"},
        {"AnswerID": 98, "QuestionId": 4, "AnswerText": "Algunos jefes comunican muy bien, otros no tanto"},
        {"AnswerID": 99, "QuestionId": 4, "AnswerText": "Recibo retroalimentación constante y útil"},
        {"AnswerID": 100, "QuestionId": 4, "AnswerText": "La comunicación depende de la situación y urgencia"},

        {"AnswerID": 101, "QuestionId": 5, "AnswerText": "Existen programas de desarrollo profesional"},
        {"AnswerID": 102, "QuestionId": 5, "AnswerText": "No hay oportunidades claras de crecimiento"},
        {"AnswerID": 103, "QuestionId": 5, "AnswerText": "Depende de mi iniciativa avanzar en la empresa"},
        {"AnswerID": 104, "QuestionId": 5, "AnswerText": "Siento que puedo mejorar y avanzar profesionalmente"},
        {"AnswerID": 105, "QuestionId": 5, "AnswerText": "No tengo claridad sobre oportunidades de desarrollo"}
    ]
}


headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)

print("Status code:", response.status_code)
print("Response JSON:", response.json())
