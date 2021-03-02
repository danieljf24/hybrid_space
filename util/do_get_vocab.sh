collection=$1
rootpath=$2

overwrite=$3
for text_style in bow rnn
do
python util/vocab.py --collection $collection --text_style $text_style --overwrite $overwrite --rootpath $rootpath
done
