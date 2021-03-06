# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 18:10:15 2021

@author: Neha
"""

# to fetch all the reviews of all the products of each page

#import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from time import sleep
import pandas as pd
import sqlite3 as sql

final_review = []

for page in range(1,250): 
    # for first 250 pages
    
    url = 'https://www.etsy.com/in-en/c/jewelry/earrings/ear-jackets-and-climbers?ref=pagination&page='+str(page)
    
    browser = webdriver.Chrome(executable_path="C:/Users/Neha/Desktop/forsk/chromedriver.exe")
    
    browser.get(url)
    
    sleep(2)
    
    
    for i in range(1,64):
        # i represent the product number. if we want 50 products, change the range to 50
        
        tabs = browser.window_handles
        browser.switch_to.window(tabs[0])
        
        path = '//*[@id="content"]/div/div[1]/div/div[3]/div[2]/div[2]/div[2]/div/div/ul/li['+str(i)+']'
        get_result = browser.find_element_by_xpath(path)
        get_result.click()
        sleep(5)
        
        tabs = browser.window_handles
        browser.switch_to.window(tabs[i])
        
        html_page = browser.page_source
        soup = BeautifulSoup(html_page)
        
        r = soup.findAll('p', {'class' : 'wt-text-truncate--multi-line wt-break-word'})
        for rev in r:
            final_review.append(rev.text.strip())
                
        # temp to find total number of pages of reviews
        temp = 0
        s = soup.findAll('span',{'class':'wt-screen-reader-only'})
        for t1 in s:
            if 'Page' in t1.text.strip() :
                if temp<int(t1.text.strip().split()[-1]):
                    temp = int(t1.text.strip().split()[-1])
                    
        for cnt in range(1,temp):
            
            page_review = browser.find_elements_by_xpath('//*[@id="reviews"]/div[2]/nav/ul/li[6]/a | //*[@id="reviews"]/div[2]/nav/ul/li[7]/a | //*[@id="reviews"]/div[2]/nav/ul/li[5]/a | //*[@id="reviews"]/div[2]/nav/ul/li[4]/a')
            #print(page_review)
            
            page_review[-1].click()
            sleep(5)
            
            tabs = browser.window_handles
            browser.switch_to.window(tabs[i])
            
            html_page = browser.page_source
            soup = BeautifulSoup(html_page)
            
            r = soup.findAll('p', {'class' : 'wt-text-truncate--multi-line wt-break-word'})
            for rev in r:
                final_review.append(rev.text.strip())
     
    

final_review = list(set(final_review))


df=pd.DataFrame(final_review,columns=['reviews'])

df.to_csv('etsy_reviews.csv', index = False)


conn = sql.connect('scrappedReviewsAll.db')
df.to_sql('scrappedReviewsAllTable', conn) 


