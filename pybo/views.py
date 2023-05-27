from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import mplfinance as mpf
import functools
from django.core.cache import cache


def index(request):
    return HttpResponse("안녕하세요 pybo에 오신 것을 환영합니다.")

def get_stock_data(code, start_date, end_date):
    cache_key = f"stock_data:{code}:{start_date}:{end_date}"
    data = cache.get(cache_key)  # 캐시에서 데이터 확인

    if data is None:
        data = fdr.DataReader(code, start_date, end_date)
        cache.set(cache_key, data)  # 데이터를 캐시에 저장

    return data


def getSymbols(market='KOSPI'):
    kospi_list = fdr.StockListing(market)  # KOSPI에 상장된 종목 리스트 가져오기
    return kospi_list.to_dict('records')

def get_symbol_name(code):
    symbol_list = getSymbols()
    for symbol in symbol_list:
        if symbol['Code'] == code:
            return symbol['Name']
    return ''


def stock_data_view(request):
    symbol_list = getSymbols()

    choices = [(symbol['Code'] + ' : ' + symbol['Name']) for symbol in symbol_list]

    if request.method == 'POST':
        choice = request.POST.get('symbol_choice')
        ndays = int(request.POST.get('ndays'))
        chart_style = request.POST.get('chart_style')
        volume_chart = bool(request.POST.get('volume_chart'))  # 체크박스 값 가져오기
    else:
        # 기본값 설정
        choice = request.GET.get('symbol_choice', '005930')
        ndays = int(request.GET.get('ndays', 50))
        chart_style = request.GET.get('chart_style', 'default')
        volume_chart = bool(request.GET.get('volume_chart'))  # 체크박스 값 가져오기

    code = choice.split(' : ')[0]
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=ndays)
    stock_data = get_stock_data(code, start_date, end_date)

    # 캔들 차트의 스타일 지정
    if chart_style == 'candle':
        mc = mpf.make_marketcolors(up='red', down='blue')
        style = mpf.make_mpf_style(marketcolors=mc)
        chart_type = 'candle'
    elif chart_style == 'line':
        style = mpf.Style()
        chart_type = 'line'
    else:
        style = chart_style
        chart_type = 'candle'

    # Volume 차트 출력 여부에 따라 설정
    if volume_chart:
        volume = True
    else:
        volume = False

    # Close 컬럼만 가져오기
    close_data = stock_data['Close']

    # 인덱스 제거
    close_data = close_data.reset_index(drop=True)

    # ExponentialSmoothing 모델 학습
    model = ExponentialSmoothing(close_data, trend='add', seasonal='add', seasonal_periods=5).fit()

    # 과거 데이터 가져오기
    past = close_data.iloc[-5:]

    # 예측 시작 인덱스와 예측 기간 설정
    n = len(close_data)
    pred_ndays = 10

    # 예측 수행
    predicted = model.predict(start=n, end=n+pred_ndays-1)
    
    # mpf.plot() 함수에서 returnfig=True로 설정하여 fig, ax 튜플 반환
    fig, ax = mpf.plot(stock_data, style=style, type=chart_type, figsize=(10, 7), volume=volume, returnfig=True)

    # Figure 객체에 접근
    fig.set_figwidth(10)
    fig.set_figheight(7)
    ax = fig.axes[0]

    # 예측 결과를 그래프에 점선으로 그리기
    joined = pd.concat([past, predicted])
    ax.plot(joined, linestyle='--', linewidth=1.5, label='ES')
    ax.legend(loc='best')
    
    lags = 20  # 자기회귀 모델의 차수
    
    
    # 데이터프레임으로 변환
    close_df = pd.DataFrame({'Close': close_data})


    # m1부터 m5까지의 컬럼 생성
    for i in range(1, 6):
        column_name = f'm{i}'
        close_df[column_name] = close_df['Close'].shift(i)
    close_df = close_df[5:]
    
    from sklearn.linear_model import LinearRegression

    # m1부터 m5까지의 컬럼과 Close 컬럼을 사용하여 데이터셋 생성
    dataset = close_df[['m1', 'm2', 'm3', 'm4', 'm5', 'Close']]

    # 학습 데이터와 타겟 데이터 분리
    X = dataset[['m1', 'm2', 'm3', 'm4', 'm5']]
    y = dataset['Close']

    # 선형 회귀 모델 학습
    model = LinearRegression()
    model.fit(X, y)
    
        # 데이터프레임에서 끝에서 5개의 Close 컬럼 데이터 선택
    ser = close_df['Close'].tail(5)

    # 예측 결과를 그래프에 추가
    n = len(ser)  # 과거 데이터의 개수
    # 예측 시작 인덱스 계산
    start_index = len(close_data) - len(ser)
    pred_index = pd.date_range(start=close_data.index[start_index], periods=pred_ndays)

    # 예측 결과를 저장할 빈 시리즈 생성
    ser_predicted = pd.Series(index=pred_index)

    for step in range(pred_ndays):
        past = pd.DataFrame(data={f'm{i}': [ser.iloc[-i - 1]] for i in range(5)})
        # 예측 수행
        predicted = model.predict(past)[0]

        # 예측한 값을 ser에 추가
        next_date = pd.to_datetime(ser.index[-1]) + timedelta(days=1)
        ser = ser.append(pd.Series([predicted], index=[next_date]))

        # 예측 결과를 ser_predicted에 추가
        ser_predicted[next_date] = predicted


    # 예측 결과를 그래프에 추가
    ax.plot(range(start_index, start_index + len(ser)), ser, linestyle='--', linewidth=1.5, label='AR(5)')

    # 범례 추가
    ax.legend(loc='best')

    
    from io import BytesIO
    import base64
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png)
    graphic = graphic.decode('utf-8')

    return render(request, 'pybo/stock_data.html', {'stock_data': stock_data, 'graphic': graphic, 'symbol_list': symbol_list, 'choices': choices, 'code': code, 'ndays': ndays, 'chart_style': chart_style})