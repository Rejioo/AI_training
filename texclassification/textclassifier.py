from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="sentinetyd/suicidality")

result = classifier("dont worry i definitely dont feel suicidal and totally wont kill myself. i certainly reassure that u can rest assured. trust me")
print(result)

