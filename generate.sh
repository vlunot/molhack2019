#!/bin/bash
src/predict.py --no-evolve
for i in {1..2}
do
    src/predict.py -s $((12321+i)) --max-length 70 --evolve-model-num 0
done
for i in {3..4}
do
    src/predict.py -s $((12321+i)) --max-length 80 --evolve-model-num 1
done
for i in {5..6}
do
    src/predict.py -s $((12321+i)) --max-length 90 --evolve-model-num 2
done
for i in {7..8}
do
    src/predict.py -s $((12321+i)) --max-length 100 --evolve-model-num 0
done
for i in {9..10}
do
    src/predict.py -s $((12321+i)) --max-length 110 --evolve-model-num 1
done
for i in {11..14}
do
    src/predict.py -s $((12321+i)) --max-length 120 --evolve-model-num 2
done
for i in {15..16}
do
    src/predict.py -s $((12321+i)) --max-length 120 --reduced-set --evolve-model-num 0
done
for i in {17..18}
do
    src/predict.py -s $((12321+i)) --max-length 120 --reduced-set --evolve-model-num 1
done
for i in {19..22}
do
    src/predict.py -s $((12321+i)) --max-length 120 --reduced-set --evolve-model-num 2
done
for i in {23..24}
do
    src/predict.py -s $((12321+i)) --max-length 120 --reduced-set --evolve-model-num 0 --num-iterations 50
done
for i in {25..26}
do
    src/predict.py -s $((12321+i)) --max-length 120 --reduced-set --evolve-model-num 1 --num-iterations 50
done
for i in {27..30}
do
    src/predict.py -s $((12321+i)) --max-length 120 --reduced-set --evolve-model-num 2 --num-iterations 50
done
for i in {31..32}
do
    src/predict.py -s $((12321+i)) --max-length 130 --evolve-model-num 0 --num-iterations 50
done
for i in {33..34}
do
    src/predict.py -s $((12321+i)) --max-length 130 --evolve-model-num 1 --num-iterations 50
done
for i in {35..36}
do
    src/predict.py -s $((12321+i)) --max-length 130 --evolve-model-num 2 --num-iterations 50
done
for i in {37..37}
do
    src/predict.py -s $((12321+i)) --max-length 130 --reduced-set --evolve-model-num 0 --num-iterations 50
done
for i in {38..38}
do
    src/predict.py -s $((12321+i)) --max-length 130 --reduced-set --evolve-model-num 1 --num-iterations 50
done
for i in {39..40}
do
    src/predict.py -s $((12321+i)) --max-length 130 --reduced-set --evolve-model-num 2 --num-iterations 50
done
for i in {41..41}
do
    src/predict.py -s $((12321+i)) --max-length 130 --evolve-model-num 0 --num-iterations 100
done
for i in {42..42}
do
    src/predict.py -s $((12321+i)) --max-length 130 --evolve-model-num 1 --num-iterations 100
done
for i in {43..43}
do
    src/predict.py -s $((12321+i)) --max-length 130 --evolve-model-num 2 --num-iterations 100
done
for i in {44..44}
do
    src/predict.py -s $((12321+i)) --max-length 130 --reduced-set --evolve-model-num 0 --num-iterations 100
done
for i in {45..45}
do
    src/predict.py -s $((12321+i)) --max-length 130 --reduced-set --evolve-model-num 1 --num-iterations 100
done
for i in {46..46}
do
    src/predict.py -s $((12321+i)) --max-length 130 --reduced-set --evolve-model-num 2 --num-iterations 100
done
