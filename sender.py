import browser_cookie3
import requests


cj = browser_cookie3.chromium()
result = requests.get("https://onti2020.ai-academy.ru/profile.php", cookies=cj)
print(result)
print(result.text)
exit()
result = requests.post("https://onti2020.ai-academy.ru/functions/submit_solution.php?bid=1&hid=3&login_redirect=https://onti2020.ai-academy.ru/profile.php",
              files=dict(
                  csv=("ensemble.json", open("tesr.json", 'rb')),
                  comment=""
              ),
              headers=dict(
                  origin="https://onti2020.ai-academy.ru",
                  referer="https://onti2020.ai-academy.ru/profile.php",
              ),
              cookies=cj)
print(result)
print(result.text)
