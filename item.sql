SELECT I_PRICE, I_NAME , I_DATA   FROM ITEM WHERE I_PRICE > 99.99
SELECT I_PRICE, I_NAME , I_DATA   FROM ITEM WHERE I_PRICE >= 99.99 AND I_ID > 8000
SELECT I_PRICE, I_NAME , I_DATA   FROM ITEM WHERE I_ID = 71909
SELECT I_PRICE, I_NAME , I_DATA   FROM ITEM WHERE I_NAME = "vugfobdihlpboqwlf"
SELECT I_PRICE, I_NAME , I_DATA   FROM ITEM WHERE I_ID = 38487
SELECT count(*)   FROM ITEM WHERE I_IM_ID < 8389 AND I_PRICE < 10.15;
SELECT count(*)   FROM ITEM WHERE I_IM_ID <= 8389 AND I_PRICE <= 10.15 ;
SELECT I_PRICE, I_NAME , I_DATA   FROM ITEM WHERE I_PRICE >= 99.99 AND I_ID > 8000 AND I_IM_ID <= 9000
SELECT count(*)   FROM ITEM WHERE I_PRICE >= 90.99 AND I_ID > 8000 AND I_IM_ID <= 9000
SELECT count(*)   FROM ITEM WHERE I_PRICE >= 90.99 AND I_ID > 8000 AND I_IM_ID <= 9000 AND I_NAME = "xpgylyfigybuxitbjitcuja"
SELECT count(*)   FROM ITEM WHERE I_NAME = "rejrgoipejodjcqmx"
SELECT count(*)  FROM ITEM WHERE I_PRICE >= 20 AND I_ID < 8000 AND I_IM_ID <= 9000
SELECT count(*)  FROM ITEM WHERE I_PRICE >= 20 AND I_ID < 1000 AND I_IM_ID <= 1200
SELECT count(*)  FROM ITEM WHERE I_PRICE >= 40 AND I_ID < 8000 AND I_IM_ID <= 9000
SELECT count(*)  FROM ITEM WHERE I_PRICE >= 70 AND I_ID < 1000 AND I_IM_ID <= 1200
