import os
import json
from io import StringIO
import subprocess
import shutil
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import speech_recognition as sr
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt
from django.template.defaulttags import register
from django.http import HttpResponse
from django.core.mail import send_mail
from django.shortcuts import render, redirect
from django.conf import settings
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
import nltk
from pydub import AudioSegment
from .newsScraper import *
from .utilityFunctions import *
from nltk.corpus import stopwords
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from docx import Document

def docxparser(data):
    doc = Document(data)
    paragraphs = [paragraph.text for paragraph in doc.paragraphs]
    text = ". ".join(paragraphs)
    with open("Output.docx.txt", "w", encoding="utf-8") as text_file:
        text_file.write(text)
    return text

def pdfparser(data):
    fp = open(data, 'rb')
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)
        data = retstr.getvalue()

    text_file = open("Output.txt", "w", encoding="utf-8")
    text_file.write(data)

    text_file = open("Output.txt", 'r', encoding="utf-8")
    a = ""
    for x in text_file:
        if len(x) > 2:
            b = x.split()
            for i in b:
                a += " "+i
    final_comment = a.split('.')
    return final_comment

def sendEmail(request):
    if request.method == 'POST':
        firstname = request.POST.get('firstname')
        lastname = request.POST.get('lastname')
        email = request.POST.get('email')
        suggestion = request.POST.get('suggestion')
        subject = f'C.E.L.T Suggestion from : {firstname}'
        message = f'First Name: {firstname}\nLast Name: {lastname}\nEmail: {email}\nSuggestion: {suggestion}'
        toEmail = ["celttsa@gmail.com"]
        send_mail(subject, message, email,toEmail, fail_silently=False)
        return render(request, 'realworld/index.html')
    else:
        return render(request, 'realworld/index.html')

def format(similarityScore):
    result_dict = {}
    total = similarityScore[0].item() + similarityScore[1].item() + similarityScore[2].item()
    result_dict['pos']=similarityScore[0].item()/total
    result_dict['neg']=similarityScore[1].item()/total
    result_dict['neu']=similarityScore[2].item()/total
    return result_dict

def imageAnalysis(request):
    if request.method == 'POST':
        imgfile = request.FILES['imageFile']
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        image = Image.open(imgfile)
        text = ["a happy image or a good image", "a sad image or a bad image", "a blank image or a neutral image"]
        inputs = processor(text, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=inputs['pixel_values'])
            text_features = model.get_text_features(input_ids=inputs['input_ids'])
        similarity_scores = (image_features @ text_features.T).squeeze(0)
        result = format(similarity_scores)
        finalText = "As this is an image, there is no text."
        return render(request, 'realworld/results.html', {'sentiment': result, 'text': finalText})
    else:
        note = "Please upload the image you want to analyze"
        return render(request, 'realworld/imageAnalysis.html', {'note': note})

def analysis(request):
    return render(request, 'realworld/index.html')

def get_clean_text(text):
    text = removeLinks(text)
    text = stripEmojis(text)
    text = removeSpecialChar(text)
    text = stripPunctuations(text)
    text = stripExtraWhiteSpaces(text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    stop_words.add('rt')
    stop_words.add('')
    newtokens = [item for item in tokens if item not in stop_words]

    textclean = ' '.join(newtokens)
    return textclean

def detailed_analysis(result):
    result_dict = {}
    neg_count = 0
    pos_count = 0
    neu_count = 0
    total_count = len(result)

    for item in result:
        cleantext = get_clean_text(str(item))
        sentiment = sentiment_scores(cleantext)
        pos_count += sentiment['pos']
        neu_count += sentiment['neu']
        neg_count += sentiment['neg']

    total = pos_count + neu_count + neg_count
    result_dict['pos'] = pos_count/total
    result_dict['neu'] = neu_count/total
    result_dict['neg'] = neg_count/total

    return result_dict

def input(request):
    if request.method == 'POST':
        file = request.FILES['document']
        fs = FileSystemStorage()
        fs.save(file.name, file)
        pathname = 'sentimental_analysis/media/'
        extension_name = file.name
        extension_name = extension_name.split('.')[-1]
        path = pathname+file.name
        destination_folder = 'sentimental_analysis/media/document/'
        shutil.copy(path, destination_folder)
        useFile = destination_folder+file.name
        result = {}
        finalText = ''
        print(extension_name)
        if extension_name == 'pdf':
            value = pdfparser(useFile)
            result = detailed_analysis(value)
            finalText = ". ".join(value)
        elif extension_name == 'txt':
            text_file = open(useFile, 'r', encoding="utf-8")
            a = ""
            for x in text_file:
                if len(x) > 2:
                    b = x.split()
                    for i in b:
                        a += " " + i
            final_comment = a.split('.')
            text_file.close()
            finalText = ". ".join(final_comment)
            result = detailed_analysis(final_comment)
        elif extension_name == 'docx':
            text = docxparser(useFile)
            final_comment = text.split('.')
            finalText = ". ".join(final_comment)
            result = detailed_analysis(final_comment)
        folder_path = 'sentimental_analysis/media/'
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

        return render(request, 'realworld/results.html', {'sentiment': result, 'text': finalText})
    else:
        note = "Please Enter the Document you want to analyze"
        return render(request, 'realworld/documentAnalysis.html', {'note': note})

def productanalysis(request):
    if request.method == 'POST':
        blogname = request.POST.get("blogname", "")

        text_file = open(
            "Amazon_Comments_Scrapper/amazon_reviews_scraping/amazon_reviews_scraping/spiders/ProductAnalysis.txt", "w")
        text_file.write(blogname)
        text_file.close()

        spider_path = r'Amazon_Comments_Scrapper/amazon_reviews_scraping/amazon_reviews_scraping/spiders/amazon_review.py'
        output_file = r'Amazon_Comments_Scrapper/amazon_reviews_scraping/amazon_reviews_scraping/spiders/reviews.json'
        command = f"scrapy runspider \"{spider_path}\" -o \"{output_file}\" "
        result = subprocess.run(command, shell=True)

        if result.returncode == 0:
            print("Scrapy spider executed successfully.")
        else:
            print("Error executing Scrapy spider.")

        with open(r'Amazon_Comments_Scrapper/amazon_reviews_scraping/amazon_reviews_scraping/spiders/reviews.json', 'r') as json_file:
            json_data = json.load(json_file)
        reviews = []

        for item in json_data:
            reviews.append(item['Review'])
        finalText = ". ".join(reviews)
        finalText = finalText.replace('\n', '\\n')
        print(finalText)
        result = detailed_analysis(reviews)
        return render(request, 'realworld/results.html', {'sentiment': result, 'text' : finalText})
    else:
        note = "Please Enter the product blog link for analysis"
        return render(request, 'realworld/productAnalysis.html', {'note': note})

def textanalysis(request):
    if request.method == 'POST':
        text_data = request.POST.get("textField", "")
        final_comment = text_data.split('.')
        result = detailed_analysis(final_comment)
        finalText = ". ".join(final_comment)
        return render(request, 'realworld/results.html', {'sentiment': result, 'text' : finalText})
    else:
        note = "Enter the Text to be analysed!"
        return render(request, 'realworld/textAnalysis.html', {'note': note})

def audioanalysis(request):
    if request.method == 'POST':
        file = request.FILES['audioFile']
        fs = FileSystemStorage()
        fs.save(file.name, file)
        pathname = "sentimental_analysis/media/"
        extension_name = file.name
        extension_name = extension_name[len(extension_name)-3:]
        path = pathname+file.name
        result = {}
        destination_folder = 'sentimental_analysis/media/audio/'
        shutil.copy(path, destination_folder)
        useFile = destination_folder+file.name
        audio = AudioSegment.from_file(useFile)
        audio = audio.set_sample_width(2)
        audio = audio.set_frame_rate(44100)
        audio = audio.set_channels(1)
        audio.export(useFile, format='wav')
        text = speech_to_text(useFile)
        final_comment = text.split('.')
        finalText = ". ".join(final_comment)
        result = detailed_analysis(final_comment)
        folder_path = 'sentimental_analysis/media/'
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        return render(request, 'realworld/results.html', {'sentiment': result, 'text' : finalText})
    else:
        note = "Please Enter the audio file you want to analyze"
        return render(request, 'realworld/audioAnalysis.html', {'note': note})


def livespeechanalysis(request):
    if request.method == 'POST':
        my_file_handle = open('./sentimental_analysis/realworld/recordedAudio.txt')
        audioFile = my_file_handle.read()
        result = {}
        text = speech_to_text(audioFile)
        final_comment = text.split('.')
        finalText = ". ".join(final_comment)
        result = detailed_analysis(final_comment)
        folder_path = 'sentimental_analysis/media/recordedAudio/'
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        return render(request, 'realworld/results.html', {'sentiment': result, 'text' : finalText})
    else:
        note = "Please record another file"
        return render(request, 'realworld/speechAnalysis.html', {'note': note})

@csrf_exempt
def recordaudio(request):
    if request.method == 'POST':
        audio_file = request.FILES['liveaudioFile']
        fs = FileSystemStorage()
        fs.save(audio_file.name, audio_file)
        folder_path = 'sentimental_analysis/media/'
        files = os.listdir(folder_path)

        pathname = "sentimental_analysis/media/"
        extension_name = audio_file.name
        extension_name = extension_name[len(extension_name)-3:]
        path = pathname+audio_file.name
        audioName = audio_file.name
        destination_folder = 'sentimental_analysis/media/recordedAudio/'
        shutil.copy(path, destination_folder)
        useFile = destination_folder+audioName
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

        audio = AudioSegment.from_file(useFile)
        audio = audio.set_sample_width(2)
        audio = audio.set_frame_rate(44100)
        audio = audio.set_channels(1)
        audio.export(useFile, format='wav')

        text_file = open("sentimental_analysis/realworld/recordedAudio.txt", "w")
        text_file.write(useFile)
        text_file.close()
        response = HttpResponse('Success! This is a 200 response.', content_type='text/plain', status=200)
        return response

def newsanalysis(request):
    if request.method == 'POST':
        topicname = request.POST.get("topicname", "")
        scrapNews(topicname)

        with open('news.json', 'r') as json_file:
            json_data = json.load(json_file)
        news = []
        for item in json_data:
            news.append(item['Summary'])

        finalText = ". ".join(news)
        finalText = finalText.replace('\n', '\\n')
        print(finalText)
        result = detailed_analysis(news)
        print(result)
        return render(request, 'realworld/results.html', {'sentiment': result, 'text' : finalText})
    else:
        return render(request, 'realworld/newsAnalysis.html')

def speech_to_text(filename):
    r = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_data = r.record(source)
        text = r.recognize_google(audio_data)
        return text

def sentiment_analyzer_scores(sentence):
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(sentence)
    return score

@register.filter(name='get_item')
def get_item(dictionary, key):
    return dictionary.get(key, 0)
