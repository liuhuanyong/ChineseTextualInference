
from urllib import request
from lxml import etree

class Translate:
    def __init__(self):
        return

    '''获取html'''
    def get_html(self, url):
        headers = {
            'User-Agent': r'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          r'Chrome/45.0.2454.85 Safari/537.36 115Browser/6.0.3',
            'Connection': 'keep-alive'
        }
        req = request.Request(url, headers=headers)
        page = request.urlopen(req).read()
        page = page.decode('utf-8')
        return page

    '''解析翻译答案'''
    def extract_answers(self, content):
        selector = etree.HTML(content)
        answer = selector.xpath('//div[@class="in-base"]/div/div/text()')[0]
        return answer

    '''翻译主函数'''
    def translate(self, query):
        url = 'http://www.iciba.com/{}'.format(query)
        html = self.get_html(url)
        try:
            answer = self.extract_answers(html)
        except Exception as e:
            answer = query
        return answer


if __name__ == '__main__':
    handler = Translate()
    while 1:
        query = input('entere an sent to translate:')
        res = handler.translate(query)
        print(res)
