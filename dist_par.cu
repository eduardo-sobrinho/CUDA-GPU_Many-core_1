#include <stdio.h>
#include <stdlib.h>

// Macro para checagem de erro das chamadas às funções do CUDA
#define checa_cuda(result) \
    if (result != cudaSuccess) { \
        printf("%s\n", cudaGetErrorString(result)); \
        exit(1); \
    }

char *aloca_sequencia(int n) {
    char *seq;

    seq = (char *) malloc((n + 1) * sizeof(char));
    if (seq == NULL) {
        printf("\nErro na alocação de estruturas\n");
        exit(1);
    }
    return seq;
}

__global__ void inicializa_GPU(int nLinhas, int mColunas, int *a)
{
    int i; // id GLOBAL da thread

    i = blockIdx.x * blockDim.x + threadIdx.x;

    // Inicializa as colunas da 1ª linha
    if (i < mColunas) {
        a[i] = i;
    }

    // Inicializa a 1ª coluna
    if (i < nLinhas) {
       a[i * mColunas] = i;
    }
}

// Kernel executado na GPU por todas as threads de todos os blocos
__global__ void distancia_GPU(int nLinhas, int mColunas, int *a, char *s, char *r, int n, int m, int *d, int deslocamento, int deslocamentoS)
{
    int i; // id GLOBAL da thread

    i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int min, celulaDiagonal;

    int it = 0;  // Usado para andar com o índice de r
    int rodada = 0;

    while(rodada < nLinhas + mColunas - 3 && deslocamentoS + i + 1 < nLinhas) {

        // Se a thread estiver após a 1ª coluna  E  no máximo na última coluna  E  no máximo na última linha
        if (rodada - i >= 0  && rodada - i < mColunas - 1 && i + 1 < nLinhas) {
            
            // Se s[i+1] e r[it+1] forem iguais, copia o valor da diagonal; senão, copia o valor da diagonal acrescido de uma unidade
            celulaDiagonal = s[deslocamentoS + i+1] == r[it + 1] ? a[deslocamento + i*mColunas - i + rodada] : a[deslocamento + i*mColunas - i + rodada] + 1;

            // Mínimo entre a célula diagonal (já calculada) e a célula de cima (acrescida de uma unidade)
            min = celulaDiagonal < a[deslocamento + i*mColunas - i + rodada + 1] + 1 ? celulaDiagonal : a[deslocamento + i*mColunas - i + rodada + 1] + 1;

            // Mínimo entre a célula à esquerda e o mínimo anterior
            if (a[deslocamento + i*mColunas + mColunas + 1 - i + rodada - 1] + 1 < min) {
                a[deslocamento + i*mColunas + mColunas + 1 - i + rodada] = a[deslocamento + i*mColunas + mColunas + 1 - i + rodada - 1] + 1;
            } else {
                a[deslocamento + i*mColunas + mColunas + 1 - i + rodada] = min;
            }

            it++;
        }

        rodada++;

        // Sincronização de barreira entre todas as threads do BLOCO
        __syncthreads();
    }

    if (i == 0) {
        *d = a[nLinhas * mColunas - 1];
    }
}

// Programa principal
int main(int argc, char **argv) {
    int nLinhas,
    mColunas,
    nBytes,
    *d_a,    // Vetor (matriz de distância) da GPU (device)
    
    *d_dist, // Variável da GPU (device) que conterá a última célula da matriz
    h_dist;  // Valor de retorno da última célula da matriz (conterá a distância)
    
    const int N_THREADS_BLOCO = 1024;

    int n,  // Tamanho da sequência s
        m;  // Tamanho da sequência r

    char *h_s,  // Sequência s de entrada (vetor com tamanho n+1)
         *h_r,  // Sequência r de entrada (vetor com tamanho m+1)
         *d_s,
         *d_r;

    FILE *arqEntrada;  // Arquivo texto de entrada

    if(argc != 2) {
        printf("O programa foi executado com argumentos incorretos.\n");
        printf("Uso: ./dist_seq <nome arquivo entrada>\n");
        exit(1);
    }

    // Abre arquivo de entrada
    arqEntrada = fopen(argv[1], "rt");

    if (arqEntrada == NULL) {
        printf("\nArquivo texto de entrada não encontrado\n");
        exit(1);
    }

    // Lê tamanho das sequências s e r
    fscanf(arqEntrada, "%d %d", &n, &m);
    n++;
    m++;

    nLinhas = n;
    mColunas = m;
    nBytes = nLinhas * mColunas * sizeof(int);

    // Aloca vetores s e r
    h_s = aloca_sequencia(n);
    h_r = aloca_sequencia(m);

    // Lê sequências do arquivo de entrada
    h_s[0] = ' ';
    h_r[0] = ' ';
    fscanf(arqEntrada, "%s", &(h_s[1]));
    fscanf(arqEntrada, "%s", &(h_r[1]));

    // Fecha arquivo de entrada
    fclose(arqEntrada);

    
    /* Alocação de memória e checagem de erro */

    // Aloca vetor (matriz de distância) na memória global da GPU
    checa_cuda(cudaMalloc((void **)&d_a, nBytes));

    // Aloca variável (distância) na memória global da GPU
    checa_cuda(cudaMalloc((void **)&d_dist, sizeof(int)));
     
    // Aloca vetor (sequência r) na memória global da GPU
    checa_cuda(cudaMalloc((void **)&d_r, m*sizeof(char)));

    // Aloca vetor (sequência s) na memória global da GPU
    checa_cuda(cudaMalloc((void **)&d_s, n*sizeof(char)));
    
    
    cudaEvent_t d_ini, d_fim;
    cudaEventCreate(&d_ini);
    cudaEventCreate(&d_fim);
    cudaEventRecord(d_ini, 0);
    
    // Determina nBlocos em função de mColunas e N_THREADS_BLOCO
    // nBlocos = teto(mColunas / N_THREADS_BLOCO)
    int nBlocos = (mColunas + N_THREADS_BLOCO - 1) / N_THREADS_BLOCO;

    inicializa_GPU<<<nBlocos, N_THREADS_BLOCO>>>(nLinhas, mColunas, d_a);
    
    // Copia a sequência s do host para a GPU e checa se houve erro
    checa_cuda(cudaMemcpy(d_s, h_s, n*sizeof(char), cudaMemcpyHostToDevice));

    // Copia a sequência r do host para a GPU e checa se houve erro
    checa_cuda(cudaMemcpy(d_r, h_r, m*sizeof(char), cudaMemcpyHostToDevice));
    
    // Host espera GPU terminar de executar
    cudaDeviceSynchronize();

    int deslocamento = 0;   // Deslocamento de x posições/células na matriz. Ou seja, a cada iteração do while abaixo é deslocado 1 bloco de linhas na matriz
    int deslocamentoS = 0;  // Deslocamento do índice de s

    int repete = nLinhas;
    
    while (repete > 0) {
    
        // Calcula a distância de edição na GPU
        distancia_GPU<<<nBlocos, N_THREADS_BLOCO>>>(nLinhas, mColunas, d_a, d_s, d_r, n, m, d_dist, deslocamento, deslocamentoS);
        
        deslocamento += N_THREADS_BLOCO * mColunas;
        deslocamentoS += N_THREADS_BLOCO; // guarda o novo início de s[] (com o deslocamento p/ a próxima chamada do bloco)
        repete = repete - N_THREADS_BLOCO;
    }
    
    checa_cuda(cudaMemcpy(&h_dist, d_dist, sizeof(int), cudaMemcpyDeviceToHost));
    
    cudaEventRecord(d_fim, 0);
    cudaEventSynchronize(d_fim);
    float d_tempo;      // Tempo de execução na GPU em milissegundos
    cudaEventElapsedTime(&d_tempo, d_ini, d_fim);
    cudaEventDestroy(d_ini);
    cudaEventDestroy(d_fim);

    printf("%d\n", h_dist);
    printf("%.2f\n", d_tempo);
    
    // Libera vetor (matriz de distância) da memória global da GPU
    cudaFree(d_a);

    // Libera vetores da memória global da GPU
    cudaFree(d_s);
    cudaFree(d_r);

    // Libera variável da memória global da GPU
    cudaFree(d_dist);

    // Libera vetores da memória do host
    free(h_s);
    free(h_r);

    return 0;
}