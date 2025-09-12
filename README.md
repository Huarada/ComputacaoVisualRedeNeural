Neural Net 3D Parallel

Visualização em OpenGL de redes neurais com camadas paralelas (estágios e branches).
Permite carregar ativações a partir de um arquivo .txt e exibir os neurônios e conexões em 3D, com cores variando conforme intensidade.

📦 Requisitos

Windows + MinGW (ou WSL/Linux equivalente)

Bibliotecas:

GLFW3
 (dll + headers + lib)

GLAD
 (código C já incluso em src/glad.c)

OpenGL 3.3+ (placa de vídeo compatível)

Certifique-se de ter o arquivo glfw3.dll no mesmo diretório do executável.

🔨 Compilação

Dentro da pasta do projeto, rode no terminal:

g++ -std=c++17 -O2 -I include src\glad.c neural_net_3d_parallel.cpp -L lib -lglfw3dll -lopengl32 -lgdi32 -luser32 -lkernel32 -o neural3d.exe


-I include → pasta onde estão glad.h e GLFW/glfw3.h

-L lib → pasta onde está glfw3dll.lib

-lopengl32 -lgdi32 -luser32 -lkernel32 → dependências do Windows/OpenGL

Se já tiver o executável (neural3d.exe), pode pular esta etapa.

📝 Input de ativações

As ativações devem estar no arquivo ativacoes.txt.
O formato aceito é um array de arrays, por exemplo:

[[0,1,2], [[4,2,1],[1,2,4]]]


Cada camada (stage) é um array.

Dentro de um estágio podem existir branches (sub-arrays).

Cada número representa a ativação/neuron count.

Valores negativos são convertidos para 0.

▶️ Execução

Para rodar lendo o arquivo:

neural3d.exe ativacoes.txt


Ou, via stdin (pipe):

type ativacoes.txt | neural3d.exe


Se não fornecer input, será usado o default:

[[0,1,2], [[4,2,1],[1,2,4]]]

🎨 Controles

Mouse esquerdo + arrastar → rotaciona a câmera em torno da rede

Scroll (dependendo da versão) → zoom (ajustável no código via gRadius)

Esferas = neurônios

Linhas = conexões entre estágios

Cores:

0 = cinza

1 = azul

2 = verde

3 = amarelo

4 = laranja

5+ = vermelho

neural_net_3d_parallel

🚀 Exemplo prático

Crie ativacoes.txt:

[[1,2,3], [[0,1],[2,2]], [4,1]]


Estrutura geral

O input no ativacoes.txt é um array de estágios:

[ stage0, stage1, stage2, ... ]


Cada stage pode ser:

Array de números → significa um único branch (sequência linear de neurônios).
Exemplo:

[3,2,1]


→ Um branch único com 3 neurônios, depois 2, depois 1.

Array de arrays de números → significa branches em paralelo dentro do mesmo estágio.
Exemplo:

[[3,1], [2,2]]


→ Dois branches paralelos:

Branch 0: 3 neurônios seguidos de 1

Branch 1: 2 neurônios seguidos de 2

📌 Exemplo do seu caso

Entrada:

[3, [3,1]]

Interpretação

Stage 0 → [3]
→ Uma camada única com 3 neurônios em série.

Stage 1 → [3,1] mas dentro de colchetes adicionais → significa um branch paralelo.
→ Um branch com:

camada com 3 neurônios

seguida de camada com 1 neurônio

Estrutura resultante

Linha principal (em série): 3 neurônios

Em paralelo, logo depois, surge um branch lateral com 3 neurônios → 1 neurônio.

Visualmente, o código vai colocar:

No eixo X → os estágios (stage0, stage1, …)

No eixo Y → os neurônios de cada camada

No eixo Z → cada branch paralelo dentro do mesmo estágio

neural_net_3d_parallel

🖼 Visual mental
Stage 0 (linear)     Stage 1 (paralelo)
      ●●●             Branch0: ●●● → ●
                        (3)      (1)


As conexões (linhas) ligam todos os neurônios de Stage0 para todos os neurônios de cada branch do Stage1.

🧪 Teste prático

Crie um ativacoes.txt com:

[[3], [3,1]]


E rode:

neural3d.exe ativacoes.txt


Você verá:

Primeiro bloco (X=-6) → 3 neurônios

Segundo bloco (X=0) → dois níveis em um branch paralelo: 3 neurônios → 1 neurônio.



Rode:

neural3d.exe ativacoes.txt


Uma janela abrirá mostrando a rede neural 3D com cores e conexões.
