# coding utf-8
import pandas as pd

column1 = ['buyinhour',  # 1 1시간 이내에 구매여부
           'plat_A', 'plat_B', 'plat_C', 'plat_D',  # 2~5 사용플랫폼 A~D
           'total_session',  # 6 목록 진입시점 방문 총 세션수
           'comic_hash',  # 7 작품을 나타내는 해쉬
           'privacy_1', 'privacy_2', 'privacy_3',  # 8~10 개인정보
          ]
column2 = ['comic' + str(x+1) for x in range(100)]  # 11~110 주요 작품 구매여부
column3 = ['comic_tag1',  # 111 작품태그
           'coin_needed',  # 112 구매시 필요코인
           'end',  # 113 완결 여부
          ]
column4 = ['schedule' + str(x+1) for x in range(123-114+1)]  # 114~123 스케쥴정보
column5 = ['genre' + str(x+1) for x in range(141-124+1)]  # 124~141 장르정보
column6 = ['last_episode',  # 142 마지막 에피소드 발행 시점
           'book',  # 143 단행본여부
           'comic_start',  # 144 작품 발행 시점
           'total_episode',  # 145 총 발행 에피소드
          ]

column7 = ['comic_tag' + str(x+2) for x in range(151-146+1)]  # 146~151 작품태그
column8 = ['user_tendency' + str(x+1) for x in range(167-152+1)]  # 152~167 유저성향 정보, 과거 구매시 기록

columns = column1 + column2 + column3 + column4 + column5 + column6 + column7 + column8

data = pd.read_csv('./data/lezhin_dataset_v2_training.tsv', sep='\t', header=None, names=columns )