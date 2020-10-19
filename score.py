##############################################################################
#                                                                             
#                      평가용 채점 스크립트 작성 베이스                         
#                                                                             
##############################################################################

'''
[안내사항]

안녕하십니까. 본 스크립트는 대회 참가자분들이 개발하신 모델의 일반화 성능 지표를 
측정하고 그 추이를 AIFactory 플랫폼을 통해 확인할 수 있도록 기록하는 데 사용되는 
평가 스크립트의 베이스 코드입니다. 참가자분들께서는 대회의 각 과제별로 제시된 
개발 목표를 파악하신 다음, 주어진 데이터셋을 이용하여 목표를 가장 잘 구현할 수 
있는 모델과 목표의 달성 수준을 잘 드러낼 수 있는 정량 지표를 설계하고, AIFactory 
플랫폼을 이용하여 그 진척도를 일반화 성능 지표의 추이로 기록하시면서 개발을 
진행하시게 됩니다. 대회 가이드에 설명되었듯, 모델의 지표상 성능 뿐만 아니라
성능을 측정하는데 사용할 지표 및 방식을 얼마나 타당하게 설계하였는지 또한 
심사의 중요한 평가요소가 됩니다. 플랫폼 사용법 설명과 아래 주석으로 이루어진
스크립트 활용 가이드의 내용을 꼼꼼히 익히시어 원하시는 모델을 성공적으로 
구현할 수 있도록 평가 스크립트 코드를 작성해주시기 바랍니다.  

작성을 끝낸 스크립트는 대회 태스크 페이지(http://aifactory.space/aic/task)에서
직접 생성하신 태스크의 [데이터]란 상단에 있는 [채점 스크립트 업로드] 버튼을 클릭하여
역시 목적에 맞게 생성하신 정답 파일과 함께 업로드 해주시면 됩니다. 이후 개발한 모델을 
이용하여 생성한 결과를 해당 태스크의 [제출하기]란을 통해 제출하시면 점수판에 
채점 결과와 그 추이가 기록됩니다. 

일반적인 스코어어택 형식의 경진대회와 달리 본 대회에서는 각 참가팀별로 목적에 맞는
모델과 평가방식을 설계하고 직접 스코어를 쌓아나가야 하며 그 타당성이 종합적으로
평가됩니다. 따라서 플랫폼에 직접 생성하신 태스크와 정답 파일, 채점 스크립트는 다른
참가자들과 공유되지 않으며, 태스크별 리더보드 스코어 역시 그 수치를 타팀의 태스크와
일괄적으로 비교하는 목적으로 사용되지 않습니다.
'''


'''
[스크립트 구조]
본 대회에 사용되는 Python 평가 스크립트는 다음과 같은 구성으로 이루어져 있습니다.

Ⅰ. 라이브러리 로드: 입출력을 포함한 전체 과정에 사용할 라이브러리를 불러오는 
                    구간입니다.

Ⅱ. 인자 정의 및 데이터 불러오기: 평가에 필요한 요소들을 입력받기 위한 인자들을 
                                정의하는 구간입니다.

Ⅲ. 전처리: 불러온 데이터를 평가용 함수에 입력하기 알맞은 형태로 재구성하는
            코드를 작성하는 구간입니다.
 
Ⅳ. 평가용 함수 정의: 평가 결과를 정량화된 수치로 반환할 함수입니다.
                     이때, 사용하실 지표가 값이 작을수록 좋은 값인지, 클수록
                     좋은 값인지에 따라 결과 정렬 방식이 달라지게 되므로
                     플랫폼에서 태스크 개설 시 정렬 기준 설정에 유의하시길
                     바랍니다.
 
Ⅴ. 평가 수행 및 스코어 출력: Ⅳ에서 정의한 평가용 함수에 Ⅲ에서 전처리한 값을 
                            입력하여 결과 스코어를 산출하는 함수입니다.

Ⅵ. 스코어 출력: 스코어를 지정한 형식으로 print하는 부분입니다. 

가장 단순하게 설명하자면, 테스트셋 정답과 추론 결과를 입력받아 스코어를 출력하게끔
작성하면 됩니다.

본 스크립트는 RMSE(roor mean squared error) 채점 예시입니다.
각 구성요소에 대한 상세 설명은 이하 스크립트 본문의 주석에 안내되어 있으니
부문별로 참고 부탁드립니다.
'''



##############################################################################
#                           Ⅰ. 라이브러리 로드
##############################################################################
import torch
import os
import numpy as np
from model import VideoLearner
from dataset import VideoRecord, VideoDataset
##############################################################################



##############################################################################
#                       Ⅱ. 인자 정의 및 데이터 불러오기                         
##############################################################################
'''
    위에 설명하였듯 본 스크립트는 메인 프로세스에서 subprocess 기능을 이용하여
    터미널로 실행됩니다. 메인프로세스가 터미널에 입력하는 실행 명령의 구조는 
    다음과 같습니다.
    
    $ python score.py <model_path> <model_name> <score_path> <num_frames:32>
    
    
'''
# gt = pd.read_csv(sys.argv[1])  # 테스트셋정답(sys.argv[1])을 pandas로 불러와
#                                # gt(ground truth)라는 이름으로 할당
# pr = pd.read_csv(sys.argv[2])  # 채점할결과(sys.argv[2])를 pandas로 불러와
#                                # pr(prediction result) 인자로 할당

def get_parser():
    parser = argparse.ArgumentParser(description="AI Championship Action recognition")
    parser.add_argument(
        "--model_path",
        default="./",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--model_name",
        default="sample_010",
        metavar="FILE",
        help="Model name format should be 'name_0EE' where E is the epoch",
    )
    parser.add_argument(
        "--score_path",
        default="test.txt",
        help="The format is 'folder/video label' ",
    )
    parser.add_argument(
        "--num_frames",
        default=32,
        help="num_frames, We all train video dataset for 8 or 32 frames",
    )

    return parser

##############################################################################



##############################################################################
#                                Ⅲ. 전처리
##############################################################################
'''
    예시 전처리 내용은 pandas 데이터프레임을 단순히 numpy array로 변환한 다음
    단일 차원의 벡터로 바꾸는 것입니다. 
    직접 코드를 작성하실 때는 평가용 함수에 입력하기 위해 필요한 전처리 과정을 
    직접 설계하신 내용에 맞추어 수행하시면 됩니다. 
'''
args = get_parser()

model_dir = os.path.join(args.model_path)
test_txt = os.path.join(args.score_path)

data = VideoDataset(model_dir, train_split_file=test_txt, test_split_file=test_txt,batch_size=4,sample_length=args.num_frames)

model = VideoLearner(data, num_classes=30)
model.load(model_name=os.path.join(model_dir, args.model_name))
##############################################################################



##############################################################################
#                             Ⅳ. 평가용 함수 정의
##############################################################################
'''
    예시에서 평가에 사용할 함수는 regression 모델 적합도 평가에 사용하는 rmse입니다.
    해당 함수는 정답(y_true), 예측(y_pred)을 인자로 입력받아 score로 반환합니다.
    score는 float 값으로, 플랫폼 리더보드에는 소수점 이하 10번째 자리까지 기록됩니다. 
    
    위 사항을 참고하여 직접 사용할 함수를 작성하고 아래의 내용을 대신하면 되겠습니다.      
'''
# We implements this function in model.evaluate !!

###############################################################################



##############################################################################
#                             Ⅴ. 평가 수행
##############################################################################
'''
    전처리 데이터와 위의 평가용 함수를 이용하여 모델의 일반화 성능 지표가 될 score를
    산출합니다. 산출된 score는 반드시 print를 통해서 화면에 출력되도록 작성해주시기 
    바랍니다. 출력시에는 아래 print()문에서와 같이 'score:'라는 문자열에 이어서 
    결과 수치를 출력하면 됩니다.
    예): score: 0.975
'''

score = learner.evaluate()
##############################################################################


'''
위 내용과 같이 작성, 채점, 출력되면 본 평가 스크립트를 실행하는 메인 프로세스가 
화면에 출력된 결과를 플랫폼으로 전송하게 됩니다. 평가 과정에서 코드 오류가 발생하는
경우 해당 오류의 메세지 전체를 플랫폼상의 관련 페이지에서 확인하실 수 있습니다.
오류가 발생하실 경우 오류 메세지를 통해 원인을 파악하신 후 수정하여 플랫폼을 통해 
업로드하시면 적용됩니다. 

참가자분들의 건승을 기원합니다.

감사합니다.
'''