from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.expected_conditions import presence_of_element_located

filename = r'C:\Users\korna\Downloads\data1.txt'

if __name__ == '__main__':
    file1 = open(filename, 'r')
    content = file1.readlines()
    for s in content:
        content[content.index(s)] = s.replace('\n', '').split(' ')

    with webdriver.Chrome(r"C:\Users\korna\Documents\driver\chromedriver.exe") as driver:
        wait = WebDriverWait(driver, 10)
        driver.get("https://www.desmos.com/calculator?lang=ru")
        elm = EC.presence_of_element_located(
            (By.XPATH, '/html/body/div[2]/div[2]/div/div/div/div[1]/div/div[2]/div[1]/div[1]/div[1]/div[1]/span'))
        wait.until(elm)
        # input()
        # driver.find_element_by_tag_name('body').send_keys(Keys.TAB + Keys.TAB + Keys.TAB)
        driver.find_element_by_xpath(
            '/html/body/div[2]/div[2]/div/div/div/div[1]/div/div[2]/div[1]/div[1]/div[1]/div[1]/span').click()
        driver.find_element_by_xpath(
            '/html/body/div[2]/div[2]/div/div/div/div[1]/div/div[2]/div[1]/div[1]/div[1]/div[2]/div/div[3]').click()
        # driver.findelements_by_xpath(
        #     r'//*[@id="graph-container"]/div/div/div/div[1]/div/div[2]/div[1]/div[1]/div[1]/div[2]/div/div[3]')[
        #     0].click()
        # l.send_keys(Keys.TAB)

        i = 0

        for s in content:
            command = Keys.TAB * (15 + i)
            # driver.find_element_by_xpath("/html/body/div[2]/div[2]/div/div/div/div[1]/div/div[2]/div[3]/div/span/div/div[1]/div/div/div/div/tr[2]/div[1]").click()
            i+=1
            command += s[0] + Keys.TAB
            command += s[1] + Keys.TAB
            driver.find_element_by_xpath("/html/body").send_keys(command)
            input()
            print(i)

        print("done")
        input()
        # while True:
        #     driver.find_element_by_tag_name('body').send_keys(Keys.ARROW_DOWN)
        # input()
        # first_result = wait.until(presence_of_element_located((By.CSS_SELECTOR, "#graph-container > div > div > div > div:nth-child(1) > div > div.dcg-exppanel-container.dcg-add-shadow > div.dcg-exppanel.dcg-disable-horizontal-scroll-to-cursor > div > span > div > div.dcg-fade-container.dcg-disable-horizontal-scroll-to-cursor > div > div > div > div > tr:nth-child(2) > div:nth-child(1)")))
        # print(first_result.get_attribute("textContent"))
