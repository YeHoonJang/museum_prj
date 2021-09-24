from selenium import webdriver
import time
import pandas as pd

review_total_df = pd.DataFrame()
driver = webdriver.Chrome("C:/chromedriver.exe")

loc_dict = {
    "서울시립미술관서소문":
        {"start": "https://www.tripadvisor.co.kr/Attraction_Review-g294197-d1809989-Reviews-or",
         "end": "-Seoul_Museum_of_Art-Seoul.html"},
    "서울시립미술관남서울":
        {"start": "https://www.tripadvisor.co.kr/Attraction_Review-g294197-d4427522-Reviews-or",
         "end": "-Seoul_Museum_of_Art_Namseoul_Bunkwan-Seoul.html"}
}

location = input(f"{list(loc_dict.keys())}\n위 목록에서 검색할 장소를 입력하세요: ")
if location == "서울시립미술관서소문":
    driver = webdriver.Chrome("C:/chromedriver.exe")
    url_str = loc_dict[location]["start"]
    url_num = str(0)
    url_keyword = loc_dict[location]["end"]

    tmp_url = url_str + url_num + url_keyword
    driver.get(tmp_url)

    page_list = driver.find_element_by_css_selector(
        "#tab-data-qa-reviews-0 > div > div.bPhtn > div:nth-child(11) > div:nth-child(3) > div > div")
    page_nums = page_list.text.split(" ")[3]
    driver.close()

    for page_num in range(0, int(page_nums) + 10, 10):
        driver = webdriver.Chrome("C:/chromedriver.exe")

        url_str = loc_dict[location]["start"]
        url_num = str(page_num)
        url_keyword = loc_dict[location]["end"]

        total_url = url_str + url_num + url_keyword
        driver.get(total_url)
        time.sleep(3)

        # text와 평점이 모두 있는 리뷰 칸 태그
        titles = driver.find_elements_by_css_selector("span > span > a > div > span")
        reviews = driver.find_elements_by_css_selector(" div.pIRBV._T.KRIav > div > span")
        dates = driver.find_elements_by_css_selector("div.WlYyy.diXIH.cspKb.bQCoY")

        for date, title, review in zip(dates, titles, reviews):
            date = "-".join([i[:-1] for i in date.text.split(" ")[:-1]])

            review_df = pd.DataFrame({'date': [date], 'title': [title.text], 'review': [review.text]})
            review_total_df = pd.concat([review_total_df, review_df])
else:
    url_str = loc_dict[location]["start"]
    url_num = str(0)
    url_keyword = loc_dict[location]["end"]

    total_url = url_str + url_num + url_keyword

    driver = webdriver.Chrome("C:/chromedriver.exe")
    driver.get(total_url)
    time.sleep(3)

    # text와 평점이 모두 있는 리뷰 칸 태그
    titles = driver.find_elements_by_css_selector("span > span > a > div > span")
    reviews = driver.find_elements_by_css_selector(" div.pIRBV._T.KRIav > div > span")
    dates = driver.find_elements_by_css_selector("div.WlYyy.diXIH.cspKb.bQCoY")

    for date, title, review in zip(dates, titles, reviews):
        date = "-".join([i[:-1] for i in date.text.split(" ")[:-1]])

        review_df = pd.DataFrame({'date': [date], 'title': [title.text], 'review': [review.text]})
        review_total_df = pd.concat([review_total_df, review_df])

review_total_df.to_csv(f"trip_advisor_{location}.csv", index=False)