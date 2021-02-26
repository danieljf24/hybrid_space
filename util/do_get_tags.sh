collection=$1
rootpath=$2

overwrite=0
th4vl=1
th4sl=0

python util/get_concept_tags.py --rootpath $rootpath --collection $collection --th4vl $th4vl --th4sl $th4sl --use_lemma --overwrite $overwrite
