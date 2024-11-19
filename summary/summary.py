from transformers import BartForConditionalGeneration, BartTokenizer

model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def summarize_text(text, max_length=130, min_length=30, length_penalty=2.0, num_beams=4):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs, 
        max_length=max_length, 
        min_length=min_length, 
        length_penalty=length_penalty, 
        num_beams=num_beams, 
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    input_text = input('enter text')
    summary = summarize_text(input_text)
    print("Summary:")
    print(summary)

