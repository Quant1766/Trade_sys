# -*- coding: utf-8 -*-
import os
import sys
import time
import datetime
import logging.config
import hashlib
import inspect
import urllib

import pandas as pd
import numpy as np


import requests


from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup as bs
from lxml import html


import json
import matplotlib
matplotlib.use("Qt5Agg")



import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import psycopg2

import quandl
import matplotlib.dates as mdates

from matplotlib import style

from dateutil.parser import parse

import hmac
from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5

from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)


from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5

from matplotlib.figure import Figure


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        layout = QtWidgets.QVBoxLayout(self._main)

        static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(static_canvas)
        self.addToolBar(NavigationToolbar(static_canvas, self))

        dynamic_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(dynamic_canvas)
        self.addToolBar(QtCore.Qt.BottomToolBarArea,
                        NavigationToolbar(dynamic_canvas, self))

        self._static_ax = static_canvas.figure.subplots()
        t = np.linspace(0, 10, 501)
        self._static_ax.plot(t, np.tan(t), ".")

        self._dynamic_ax = dynamic_canvas.figure.subplots()
        self._timer = dynamic_canvas.new_timer(
            100, [(self._update_canvas, (), {})])
        self._timer.start()

    def _update_canvas(self):
        self._dynamic_ax.clear()
        t = np.linspace(0, 10, 101)
        # Shift the sinusoid as a function of time.
        self._dynamic_ax.plot(t, np.sin(t + time.time()))
        self._dynamic_ax.figure.canvas.draw()




class Marketcap_parser(object):

    def __init__(self, **kwargs):
        default_attr = dict(start_csv_scrape=0, fin_csv_scrape=20, verbose=0, prox='207.229.93.66:1028')

        allowed_attr = list(default_attr.keys())
        default_attr.update(kwargs)

        for key in default_attr:
            if key in allowed_attr:
                self.__dict__[key] = default_attr.get(key)

        self.currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.parentdir = os.path.dirname(self.currentdir)

        self.logger = Marketcap_parser.get_logger(level=logging.DEBUG, verbose=self.verbose)

        try:
        
            engine = create_engine("postgresql://user:" + 'pass' + "@host/table")
        
            data_df = pd.read_sql_query('select * from tbl_fuel_master', con=engine)
        
        except EOFError as error_db:
            self.logger.info("Can`t get data from databaSE {0}".format(error_db))
        
            self.logger.exception("Can`t get data from databaSE {0}".format(error_db))

        self.engine = create_engine("postgresql://localhost:5432/postgres")

    def path_to(self, dir_name, file_name):
        return os.path.join(os.path.join(self.currentdir, dir_name), file_name)

    def parse_string_date(self, str_date):
        return str(parse(str_date)).split()[0]

    def get_currencies_coinmarketcap(self, url_coins="https://coinmarketcap.com/coins/views/all/"):

        res = requests.get(url_coins)

        print(res.url)

        htm = res.text

        tree = html.fromstring(htm)

        try:
            currencies = [i.split("/")[2] for i in
                          tree.xpath('//a[@class="currency-name-container link-secondary"]/@href')]
        except:
            currencies = ""

        currencies_ = pd.DataFrame(currencies, columns=['currencies'])

        currencies_.to_csv(self.path_to("stock_data", "handle_currency.csv"), index=False)

        return currencies

    def coinmarketcap_stock_data(self, currency, start_date="20130428",
                                 fin_date=datetime.date.today().isoformat().replace("-", ""), reindex=False,
                                 formatfile="csv"):

        start_url = 'https://coinmarketcap.com'
        url = start_url + '/currencies/{0}/historical-data/?start={1}&end={2}'.format(currency, start_date, fin_date)

        res = requests.get(url)

        htm = res.text

        tree = html.fromstring(htm)

        data = {}

        try:
            data["Date"] = [str(parse(date.strip())).split()[0] for date in tree.xpath('//tbody//tr//td[1]//text()') if
                            date.strip()]
        except:
            data["Date"] = ""

        try:
            data["Open"] = tree.xpath('//tbody//tr//td[2]//text()')
        except:
            data["Open"] = ""

        try:
            data["High"] = tree.xpath('//tbody//tr//td[3]//text()')
        except:
            data["High"] = ""

        try:
            data["Low"] = tree.xpath('//tbody//tr//td[4]//text()')
        except:
            data["Low"] = ""

        try:
            data["Close"] = tree.xpath('//tbody//tr//td[5]//text()')
        except:
            data["Close"] = ""

        try:
            data["Market Cap"] = tree.xpath('//tbody//tr//td[7]//text()')
        except:
            data["Market Cap"] = ""

        try:
            data["Volume"] = tree.xpath('//tbody//tr//td[6]//text()')
        except:
            data["Volume"] = ""

        dataframe = pd.DataFrame(data)

        if reindex:
            dataframe = self.reindex_dataframe(dataframe)

        dataframe.sort_values(by='Date')

        if formatfile.upper() is 'CSV':
            dataframe.to_csv(self.path_to("stock_data", "{0}.csv".format(currency)),  # ,start_date,fin_date)),
                             index=False)

        if formatfile.upper() is 'JSON':
            dataframe.to_json(self.path_to("stock_data", "{0}.json".format(currency)),  # ,start_date,fin_date)),
                              )

        return dataframe

    def reindex_dataframe(self, dataframe):

        return dataframe.iloc[::-1]

    def stockreader_from_file(self, currencyname):

        srock_data = pd.read_csv(self.path_to("stock_data", "{0}.csv".format(currencyname)))

        return srock_data

    def rewrite_to_database(self, stockframe, tablename, engine=None):
        if engine is None:
            engine = self.engine

        stockframe.to_sql(tablename.upper(), engine, if_exists="replace")

    def visualisation_data(self, currency_name, date_ftom="20130428", date_to=datetime.date.today().isoformat(),
                           stock=None):
        if stock is None:
            stock = pd.read_csv(self.path_to("stock_data",
                                             currency_name + ".csv"))  # ,index_col=["Close","Date","High","Low","Open","Volume"])
        style.use("ggplot")
        stock.index = pd.to_datetime(stock.Date)
        stock = stock.loc[date_ftom:date_to, :]
        stock["Close"].plot()
        plt.show()

    def NewTable(self, currency_name):
        cur = self.database_conect.cursor()

        cur.execute('''CREATE TABLE {0}
        (
          Date CHAR(250) NOT NULL
            CONSTRAINT {0}_pkey
            PRIMARY KEY,
          Open   INT ,
          High     INT,
          Low INT,
          Close  CHAINT,
          Market_Cap  INT,
          Volume  INT
        );'''.format(currency_name))

    def dataBaseWritbase(self, currency_name, data):

        try:
            self.logger.info("Write to database {0}".format(data))

            con = psycopg2.connect(dbname="dbname", user="user", host='host',
                                   password='password')
            cur = con.cursor()
            cur.execute("ROLLBACK")
            cur.executemany('''INSERT INTO {0} ( Date,   Open,   High,   Low,   Close,   Market_Cap,Volume )
                                    VALUES    ( %(Date)s,   %(Open)d,   %(High)d,   %(Low)d,   %(Close)d,   %(Market_Cap)d, %(Volume)d )
                                    ON CONFLICT("Date") DO UPDATE SET
                                    Open = EXCLUDED.Open,
                                    High = EXCLUDED.High,
                                    Low = EXCLUDED.Low,
                                    Close = EXCLUDED.Close, 
                                    Market_Cap = EXCLUDED.Market_Cap,
                                    Volume = EXCLUDED.Volume;'''.format(currency_name), data)

            con.commit()

            self.logger.info("Committed")

        except Exception as e:
            self.logger.exception(e)
            return e

    def update_all_cryptocurrency_coinmarketcap(self, year, month, day):
        names = pd.read_csv(self.path_to("stock_data", "handle_currency.csv"))
        startdata_ = ''.join(str(datetime.datetime(year, month, day)).split()[0].split("-"))

        fin_date = ''.join(str(datetime.date.today()).split()[0].split("-"))

        # print(names[:4])
        for name in names["currencies"][:4]:
            print(name)

            self.coinmarketcap_stock_data(name, start_date=startdata_, fin_date=fin_date)

    def get_stock_quantl(self, currency="AAPL", stock="WIKI/", start=datetime.datetime(2016, 1, 1),
                         end=datetime.date.today()):

        stock_h_data = quandl.get(stock + currency, start_date=start, end_date=end)

        # type(apple)

        apple = pd.DataFrame(stock_h_data)

        apple.to_csv(self.path_to("stock_data", currency + ".csv"))

    @staticmethod
    def get_logger(level=logging.DEBUG, verbose=0):
        logger = logging.getLogger(__name__)
        if logger.handlers == []:
            fh = logging.FileHandler('./trade_sys.log', 'a')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            fh.setLevel(level)
            logger.addHandler(fh)

            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            sh_lvls = [logging.ERROR, logging.WARNING, logging.INFO]
            sh.setLevel(sh_lvls[verbose])
            logger.addHandler(sh)

            logger.setLevel(level)

        return logger


class Exmo_trade_bot_browserversion(object):

    def __init__(self,*kwargs):
        default_attr = dict(login_exmo="<login_exmo>", pass_exmo="<pass_exmo>", verbose=0, prox='host:port')

        allowed_attr = list(default_attr.keys())
        default_attr.update(kwargs)

        for key in default_attr:
            if key in allowed_attr:
                self.__dict__[key] = default_attr.get(key)

        self.currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.parentdir = os.path.dirname(self.currentdir)

        self.logger = Exmo_trade_bot_browserversion.get_logger(level=logging.DEBUG, verbose=self.verbose)
        self.login_url = "https://exmo.com/en/login"

        self.init_driver()
        self.autorisation()


    def init_driver(self):
        #options = Options()
        options = webdriver.ChromeOptions()

        #options.add_experimental_option("prefs", {'profile.managed_default_content_settings.javascript': 2})

        # options.add_argument('headless')
        # driver = webdriver.Chrome(chrome_options=options)


        try:
            self.display = Display(visible=0, size=(800, 600))
            self.display.start()
            self.logger.info("Display started")
        except Exception as e :
            self.logger.info("Display error (0)".format(e))


        self.logger.info("Driver init")
        try:
            print ('here1')
            #chrome_options = ChromeOptions()

            self.logger.info("Driver 1")
            # chrome_options.add_argument("--headless")
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            #options.add_argument("--disable-javascript")
            self.logger.info("Driver 2")
            self.driver = webdriver.Chrome(chrome_options=options)
            self.logger.info("Driver inited")
        except Exception as error:
            self.logger.error("Driver {0}".format(error))

    def autorisation(self):
        try:
            self.logger.info("Autorizate")
            LoginEmail = self.login_exmo
            LoginPass = self.pass_exmo

            self.driver.get(self.login_url)

            self.driver.create_options()

            time.sleep(8)

            captcha_ = self.driver.find_element_by_xpath('//iframe').get_attribute("src").split("&k=")[1].split("&co=")[0]
            print("captcha_",captcha_)


            x_try = 0

            while True:

                try:
                    wait = WebDriverWait(self.driver, 10).until(EC.presence_of_element_located(
                        (By.XPATH, '//input[contains(@class,"reg_form_login")]')))
                    break
                except:

                    x_try += 1
                    print(("load wait {0}".format(x_try)))
                    self.logger.info("load wait {0}".format(x_try))
                    if x_try > 10:
                        self.terminate()

            print('here2')
            Logn = self.driver.find_element_by_xpath('//input[contains(@class,"reg_form_login")]')
            Logn.clear()
            Logn.send_keys(LoginEmail)

            psw = self.driver.find_element_by_xpath('//input[contains(@class,"reg_form_pass")]')
            psw.clear()
            psw.send_keys(LoginPass)

            self.re_captcha2(captcha_)

            self.driver.find_element_by_xpath('//button[contains(@class,"btn btn_medium") and contains(.,"Sign in")]').click()
            time.sleep(1)

            self.driver.switch_to_default_content()

            self.driver.get('https://exmo.com/en/login')

            self.logger.info("Autorizated")
            print('here3')

        except Exception as err:
            self.logger.exception(err)



    def re_captcha2(self,capcha_key):
        self.logger.info('google ReCaptcha 2')
        API_KEY = '<API_KEY>'  # Your 2captcha API KEY

        url = 'https://exmo.com/en/login'  # example url

        s = requests.Session()

       
        captcha_id = s.post(
            "https://2captcha.com/in.php?key={0}&method=userrecaptcha&googlekey={1}&pageurl={2}".format(API_KEY, capcha_key,
                                                                                                  url)).text.split('|')[1]

        print("captcha_id",captcha_id)

        recaptcha_answer = s.get("https://2captcha.com/res.php?key={0}&action=get&id={1}".format(API_KEY, captcha_id)).text
        print("recaptcha_answer", recaptcha_answer)
        while 'CAPCHA_NOT_READY' in recaptcha_answer:
            time.sleep(5)
            recaptcha_answer = s.get("https://2captcha.com/res.php?key={0}&action=get&id={1}".format(API_KEY, captcha_id)).text
        if 'ERROR_CAPTCHA_UNSOLVABLE' in recaptcha_answer:
            self.logger.error('ERROR_CAPTCHA_UNSOLVABLE')
            return 0

        self.logger.info("recaptcha_answer {0}".format( recaptcha_answer))

      

        recaptcha_answer = recaptcha_answer#.split('|')[1]
        #recaptcha_answer = recaptcha_answer[1:]

        print("recaptcha_answer", recaptcha_answer)


        payload_ = {'g-recaptcha-response': '|'.join(recaptcha_answer)}

        url_for_captcha  = "https://www.google.com/recaptcha/api2/anchor?ar=1&k={}".format(capcha_key)
    

        response = s.post(url_for_captcha, payload_)

        if len(response.cookies.get_dict())>1:
            self.logger.info('Recaption')




    @staticmethod
    def get_logger(level=logging.DEBUG, verbose=0):
        logger = logging.getLogger(__name__)
        if logger.handlers == []:
            fh = logging.FileHandler('./trade_sys.log', 'a')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            fh.setLevel(level)
            logger.addHandler(fh)

            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            sh_lvls = [logging.ERROR, logging.WARNING, logging.INFO]
            sh.setLevel(sh_lvls[verbose])
            logger.addHandler(sh)

            logger.setLevel(level)

        return logger

class VisualizationStockData(object):
    def __init__(self, **kwargs):
        default_attr = dict(StockData=None, method='plot')

        allowed_attr = list(default_attr.keys())
        default_attr.update(kwargs)

        for key in default_attr:
            if key in allowed_attr:
                self.__dict__[key] = default_attr.get(key)

        self.currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.parentdir = os.path.dirname(self.currentdir)

    def plot(self):
        pass


class ExmoAPI(object):
    def __init__(self, API_KEY, API_SECRET, API_URL='http://api.exmo.com', API_VERSION='v1'):
        self.API_URL = API_URL
        self.API_VERSION = API_VERSION
        self.API_KEY = API_KEY
        self.API_SECRET = bytes(API_SECRET, encoding='utf-8')

    def sha512(self, data):
        H = hmac.new(key=self.API_SECRET, digestmod=hashlib.sha512)
        H.update(data.encode('utf-8'))
        return H.hexdigest()

    def api_query(self, api_method, params={}):
        params['nonce'] = int(round(time.time() * 1000))
        params = urllib.parse.urlencode(params)

        sign = self.sha512(params)
        headers = {
            "Content-type": "application/x-www-form-urlencoded",
            "Key": self.API_KEY,
            "Sign": sign
        }
        url = "/".join([self.API_URL, self.API_VERSION, api_method])
        response = requests.post(url, data=params, headers=headers)

        try:
            obj = json.loads(response.text)
            if 'error' in obj and obj['error']:
                print(obj['error'])
                raise sys.exit()
            return obj
        except json.decoder.JSONDecodeError:
            print('Error while parsing response:', response)
            raise sys.exit()

    # return json for trades operation
    def get_trades(self, bin_currency):
        """trade_id - deal identifier
            type - type of the deal
            price - deal price
            quantity - currency quantity
            amount - total sum of the deal
            date - date and time of the deal Unix
            """
        bin_currency = ','.join(bin_currency)
        url = "/".join([self.API_URL, self.API_VERSION, "trades", "?pair={0}".format(bin_currency)])
        res = requests.get(url)
        if res.status_code is 200 and len(res.text) > 5:
            res_json = json.loads(res.text)
            res_json['Error'] = "None"
            return res_json
        else:
            return json.loads('{"Error":"Invalid currency name"}')

    def get_order_book(self, pair, limit=100):

        """ask_quantity - the sum of all quantity values in sell orders
            ask_amount - the sum of all total sum values in sell orders
            ask_top - minimum sell price
            bid_quantity - the sum of all quantity values in buy orders
            bid_amount - the sum of all total sum values in buy orders
            bid_top - maximum buy price
            bid - the list of buy orders where every field is: price, quantity and amount
            ask - the list of sell orders where every field is: price, quantity and amount"""

        url = "/".join([self.API_URL, self.API_VERSION, "order_book", "?pair={0}&limit={1}".format(pair, str(limit))])
        order_book_ = requests.get(url)
        if order_book_.status_code is 200 and len(order_book_.text) > 5:
            res_json = json.loads(order_book_.text)
            res_json['Error'] = "None"
            return res_json
        else:
            return json.loads('{"Error":"Invalid currency name"}')

    def get_pair_requires(self):
        """min_quantity - minimum quantity for the order
           max_quantity - maximum quantity for the order
           min_price - minimum price for the order
           max_price - maximum price for the order
           min_amount - minimum total sum for the order
           max_amount - maximum total sum for the order"""

        url = "/".join([self.API_URL, self.API_VERSION, "pair_settings"])

        pair_requires_ = requests.get(url)
        if pair_requires_.status_code is 200:
            currencys = json.loads(pair_requires_.text)
            currencys['Error'] = "None"
            return currencys

        else:
            return json.loads('{"Error":"Cant connect to exchange"}')

    def get_ticker(self):
        """high - maximum deal price within the last 24 hours
            low - minimum deal price within the last 24 hours
            avg - average deal price within the last 24 hours
            vol - the volume of deals within the last 24 hours
            vol_curr - the total value of all deals within the last 24 hours
            last_trade - last deal price
            buy_price - current maximum buy price
            sell_price - current minimum sell price
            updated - date and time of data update"""

        url = "/".join([self.API_URL, self.API_VERSION, "ticker"])

        ticker_ = requests.get(url)
        if ticker_.status_code is 200:
            currencys = json.loads(ticker_.text)
            currencys['Error'] = "None"
            return currencys

        else:
            return json.loads('{"Error":"Cant connect to exchange"}')

    def get_currencys(self):

        url = "/".join([self.API_URL, self.API_VERSION, "currency"])

        currencys_ = requests.get(url)
        if currencys_.status_code is 200:
            currencys = {"currencys": json.loads(currencys_.text)}
            currencys['Error'] = "None"
            return currencys

        else:
            return json.loads('{"Error":"Cant connect to exchange"}')

    def order_create(self, pair="STQ_USD", quantity=1, price=1, type_="buy"):
        """pair - валютная пара

        quantity - кол-во по ордеру

        price - цена по ордеру

        type - тип ордера, может принимать следующие значения:

        buy - ордер на покупку
        sell - ордер на продажу
        market_buy - ордера на покупку по рынку
        market_sell - ордер на продажу по рынку
        market_buy_total - ордер на покупку по рынку на определенную сумму
        market_sell_total - ордер на продажу по рынку на определенную сумму
        """
        params = {'pair': pair,
                  "quantity": quantity,
                  'price': price,
                  "type": type_}
        resp = self.api_query('order_create', params=params)
        return resp

    def cancle_order(self, order_id):

        resp = self.api_query('order_cancel', params={'order_id': order_id})

        return resp

    def user_open_orders(self):
        resp = self.api_query('user_open_orders')
        return resp

    def user_trades(self, pair, offset=0, limit=100):
        """pair - одна или несколько валютных пар разделенных запятой (пример BTC_USD,BTC_EUR)

            offset - смещение от последней сделки (по умолчанию 0)

            limit - кол-во возвращаемых сделок (по умолчанию 100, максимум 10 000)"""
        params = {'pair': pair,
                  "offset": offset,
                  'limit': limit}
        resp = self.api_query('user_trades', params=params)

        """trade_id - идентификатор сделки

            date - дата и время сделки

            type - тип сделки

            pair - валютная пара

            order_id - идентификатор ордера пользователя

            quantity - кол-во по сделке

            price - цена сделки

            amount - сумма сделки"""

        return resp

    def user_cancelled_orders(self, offset=0, limit=100):

        """offset - смещение от последней сделки (по умолчанию 0)

            limit - кол-во возвращаемых сделок (по умолчанию 100, максимум 10 000)"""

        params = {'offset': offset,
                  "limit": limit}

        resp = self.api_query('user_cancelled_orders', params=params)

        """date - дата и время отмены ордера

            order_id - идентификатор ордера

            order_type - тип ордера

            pair - валютная пара

            price - цена по ордеру

            quantity - кол-во по ордеру

            amount - сумма по ордеру

            """

        return resp

    def order_trades(self, order_id):
        resp = self.api_query('order_trades', params={'order_id': order_id})

        """type - тип ордера

            in_currency - валюта входящая

            in_amount - кол-во входящей валюты

            out_currency - валюта исходящая

            out_amount - кол-во исходящей валюты

            trades - массив сделок, где:

            trade_id - идентификатор сделки
            date - дата сделки
            type - тип сделки
            pair - валютная пара
            order_id - идентификатор ордера
            quantity - кол-во по сделке
            price - цена по сделке
            amount - сумма по сделке"""

        return resp

    def required_amount(self, pair="BTC_USD", quantity=1):

        """pair - валютная пара

            quantity - кол-во которое необходимо купить"""

        params = {'pair': pair,
                  "quantity": quantity}

        resp = self.api_query('required_amount', params=params)
        return resp

    def deposit_address(self):
        resp = self.api_query('deposit_address')
        return resp

class Algoritms(object):


    def ExpMovingAverage(self,values, window=5,mode='full'):
        weights = np.exp(np.linspace(-1., 0., window))

        weights /= weights.sum()
        a = np.convolve(values, weights, mode=mode)[:len(values)]
        a[:window] = a[window]
        return a
    
    def movingaverage(values, window):
    weigths = np.repeat(1.0, window) / window
    smas = np.convolve(values, weigths, 'valid')
    return smas

    def AU_AD(self,data_stock):

        data_stock = np.array(data_stock)
        AU = []
        AD = []
        AU_ = []
        AD_ = []
        for t, price in enumerate(data_stock):
            if t > 1:
                res = price - data_stock[t - 1]
                if res > 0:
                    AU.append(res)
                    AU_.append(price)
                elif res < 0:
                    AD.append(res)
                    AD_.append(price)
                else:
                    pass
                
        # AD = sum(AD_)/len(AD)
        # AU = sum(AU_)/ len(AU)
        AD = np.array(AD)
        AU = np.array(AU)
        return AU, AD

    def rsiFunc(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n + 1]
    up = seed[seed >= 0].sum() / n
    down = -seed[seed < 0].sum() / n
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100. / (1. + rs)

    for i in range(n, len(prices)):
        delta = deltas[i - 1]  # cause the diff is 1 shorter

        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (n - 1) + upval) / n
        down = (down * (n - 1) + downval) / n

        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)

    return rsi
