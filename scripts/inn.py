from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# Задаем опции для браузера
options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--incognito')
# options.add_argument('--headless')

# Путь к драйверу браузера
driver_path = '/driver/chromedriver.exe'

# Инициализация драйвера
driver = webdriver.Chrome(driver_path, options=options)

# Адрес страницы для обкачки


# Массив для хранения адресов ИНН
inn_list = []

# Обкачиваем страницы с адресами ИНН
for i in range(1, 21):
    url = f'https://inndex.ru/ul/stupino?show-active=1&page={i}'
    driver.get(url)
    # Ждем загрузки элементов на странице
    time.sleep(5)
    # Получаем все строки таблицы с адресами ИНН
    rows = driver.find_elements(By.XPATH, '//div[@class="row item-list space company-row control-over"]')

    for row in rows:
        # Получаем адрес ИНН и добавляем его в массив
        inn_list.append(row.text.split('\n')[2])

print(inn_list)
# Выводим список адресов ИНН
with open("inn_list.txt", "w", encoding='UTF-8') as output:
    for item in inn_list:
        # write each item on a new line
        output.write("%s\n" % item)

# Закрываем драйвер
driver.quit()