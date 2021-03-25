rootpath=@@@rootpath@@@
collectionStrt=@@@collectionStrt@@@
testCollection=@@@testCollection@@@
logger_name=@@@logger_name@@@
overwrite=@@@overwrite@@@

gpu=1

CUDA_VISIBLE_DEVICES=$gpu python tester.py --testCollection $testCollection --rootpath $rootpath --overwrite $overwrite --logger_name $logger_name
