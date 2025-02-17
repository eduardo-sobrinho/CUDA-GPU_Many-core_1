# CUDA-GPU_Many-core_v1

Programação Paralela para processador Many-core (GPU) usando CUDA

Cálculo da distância de edição entre 2 sequências:<br>
• Sequências de DNA: cadeias de bases nitrogenadas (A, C, G e T)<br>
• Entradas: Sequências S e R, n = |S| e m = |R|, n ≤ m<br>
• Saída: Distância de edição entre S e R (nº mínimo de operações de edição necessárias para transformar S em R)

Para compilar: nvcc dist_par.cu -o dist_par
<br>
Para executar: ./dist_par <arquivo_de_entrada>

Neste programa é implementada a solução inicial (com grid de um único bloco)
