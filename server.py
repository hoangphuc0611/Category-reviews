from flask import Flask,render_template
from flask import request, jsonify
from flask_cors import CORS, cross_origin
import urllib.request
import joblib
from bs4 import BeautifulSoup
import re
import csv
import os
import json
import pandas as pd
from underthesea import word_tokenize
import numpy as np
from selenium import webdriver
import time
from selenium.webdriver.common.keys import Keys


# Khai báo port của server
my_port = '5000'

app = Flask(__name__)
CORS(app)

# Khai bao ham xu ly request index
@app.route('/')
@cross_origin()
def index():
    return render_template('client.html')

def load_tiki(url,driver):
    dic = {}
    lis=[]
    driver.get(url)
    dic['img']=driver.find_element_by_css_selector("div[class='thumbnail']>div>div[class='container']>img").get_attribute('src')
    dic['name'] = driver.find_element_by_css_selector("h1[itemprop='name']").text
    try:
        for i in range(2):
            driver.find_element_by_css_selector('body').send_keys(Keys.CONTROL+Keys.END)
            driver.find_element_by_css_selector('body').send_keys(Keys.CONTROL+Keys.HOME)
            driver.find_element_by_css_selector('body').send_keys(Keys.CONTROL + Keys.HOME)
            time.sleep(1.5)
    except:
        pass
    x = driver.find_elements_by_css_selector("div[class='review-comment__content']")
    y = driver.find_elements_by_css_selector("div[class='review-comment__title']")
    for i, j in zip(x, y):
        if str(i.text) != '':
            lis.append(i.text)
        else:
            lis.append(j.text)
    while driver.find_elements_by_css_selector("a[class='btn next']"):
        try:
            driver.find_element_by_css_selector("a[class='btn next']").click()
            time.sleep(0.5)
        except:
            pass
        x=driver.find_elements_by_css_selector("div[class='review-comment__content']")
        y= driver.find_elements_by_css_selector("div[class='review-comment__title']")
        for i,j in zip(x,y):
            if str(i.text)!='':
                lis.append(i.text)
            else:
                lis.append(j.text)

    dic['comment']=lis
    return dic
    
# Xóa icon
def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF" 
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F1E0-\U0001F1FF"  
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)


# Xử lí các câu bình luận cào về từ web
def standardize_data(row):
  
    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row =deEmojify(row)
    row = re.sub(r"[\.,\?]+$-", "", row)
    # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
    row = row.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("=", " ") \
        .replace("(", " ").replace(")", " ") \
        .replace("-", " ").replace("?", " ")

    row = row.strip()
    return row

#Tách từ
def tokenizer(row):
    return word_tokenize(row, format="text")

# Đánh giá sản phẩm dựa vào so sánh số lượng bình lượng tốt và xấu về sản phẩm
def analyze(result,name):
    bad = np.count_nonzero(result)
    good = len(result) - bad
    if good>bad:
        return 'Sản phẩm '+name+", có vẻ  đây là 1 sản phẩm tốt"
    else:
        return 'Sản phẩm '+name+", có vẻ không phải là 1 sản phẩm tốt"


@app.route('/',methods=['POST'])
def my_form_post():
    driver=webdriver.Chrome('./chromedriver_linux64/chromedriver')
    text=request.form['u']
    dic={}
    #Tiki
    dic=load_tiki(text,driver)
    print(dic)
    img = dic['img']
    data_frame = pd.DataFrame(dic['comment'])
    data_frame[0] = data_frame[0].apply(standardize_data)
    data_frame[0] = data_frame[0].apply(tokenizer)
    X_val = data_frame[0]
    print(X_val)
    emb = joblib.load('tfidf.pkl')
    X_val = emb.transform(X_val)
    model = joblib.load('model.pkl')
    result = model.predict(X_val)
    print(result)
    return render_template('client.html',text=analyze(result,dic['name']),img=img)

# Thuc thi server
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=my_port)