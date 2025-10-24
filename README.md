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

▶️ Para Compilar:
```bash
g++ -std=c++17 -O2 -I include src\glad.c .\neural_net_3d_conv_refactored.cpp -L lib -lglfw3dll -lopengl32 -lgdi32 -luser32 -lkernel32 -o neural3d.exe
```


▶️ Execução

Para rodar lendo o arquivo:

neural3d.exe ativacoes.txt
ou em windows:
```bash
.\neural3d.exe .\ativacoes.txt
```


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

Esses arquivos são temporários e utilizados apenas para compor o JSON unificado.
NOTA: O que cada parte do código faz

O projeto é single-file mas fortemente modularizado por classes, com nomes longos e comentários:

1 namespace Math

  Funções matemáticas explícitas (identidade, perspective, lookAt, normalização, produto vetorial, etc.).
  Importante para depuração: todas as matrizes/operadores têm nomes que indicam intenção (ex.: MakeLookAtMatrix, MultiplyMat4).

2 Shaders (kLineVS/kLineFS e kLitVS/kLitFS)

  Line shader: desenha as conexões entre camadas.
  
  Lit shader: Phong “barato” para iluminação difusa + rim light, usado em esferas (neurônios) e painéis (folhas de conv).

3 Geometry builders

  BuildNeuronSphereMesh(...)
  Gera esfera (pos + normal) para cada neurônio.
  
  BuildConvPanelCubeMesh(...)
  Gera um cubo unitário. Cada “folha” da CONV é um cubo escalado (painel fino).

4 Parser de entrada (AST)

  ParseNode(...) + auxiliares (ParseNumberToken, ParseConvToken etc.)
  Interpretam o texto do arquivo na mesma gramática do código original.

5 Semântica do modelo (LayerUnit, Stage, Branch)
   
   LayerUnit descreve um elemento lógico da rede:
   
   kNeuron + neuronCount (apenas para colorir/legenda pedagógica)
   
   kConvolution + A,B,C (dimensões AxBxC)
   
   Stage é um vetor de Branch.
   
   Branch é um vetor de LayerUnit.
   
   Esta estrutura preserva a ideia de múltiplos ramos por estágio.

6 Visuals (dados prontos para render)

  NeuronLayerVisual
  Guarda instâncias: centro, raio e “nível” para cor.
  
  ConvLayerVisual
  Guarda blocos {A,B,C} convertidos em dimensões no mundo (width/height/depth) e parâmetros visuais:
  
  panelSpacingMultiplier → espaço “em branco” entre as folhas (sem alterar a grossura).
  
  baseColorRGB → cor base do gradiente de profundidade.

7 CnnModelLayout

  Coração do pipeline de dados:
  
  LoadInputAndBuildLayout(argc, argv)
  
  Lê argv[1] (ou STDIN)
  
  Faz o parse para AST → parsedStages
  
  Constrói visuals + âncoras + linhas.
  
  BuildVisualLayersAndComputeBounds()
  
  Normaliza dimensões com base nos máximos {A,B,C} do modelo.
  
  Calcula a posição (X/Y/Z) de cada instância:
  
  Esferas (neurônios) → raio padrão (configurável).
  
  Blocos conv → largura/altura/profundidade proporcionais aos seus A/B/C.
  
  Gera âncoras para conexões (lado direito/esquerdo), respeitando o meio-tamanho em X:
  
  Neurônios usam o raio como halfWidthX.
  
  Convs usam metade da largura.
  
  Constrói connectionLineVertices (lista xyz intercalada) ligando todo estágio s → s+1.
  
  Getter público sugerido:
  
  GetConnectionLineVertices() retorna as linhas para o renderer.
  
  Parâmetros visuais fáceis de ajustar (comentados no código):
  
  Raio do neurônio: kNeuronRadius
  
  Espaçamento vertical (entre linhas): kRowSpacingY
  
  Espaçamento entre ramos (Z): kBranchSpacingZ
  
  Espaçamento entre folhas da CONV: ConvLayerVisual::panelSpacingMultiplier

8 SceneRenderer

 Tudo de OpenGL + câmera:

 Criação de janela/GL, compilação e link dos shaders.

 Criação dos VAOs/VBOs para esfera, cubo e linhas.
 
 Câmera orbital com callbacks (mouse/scroll).
 
 UploadConnectionLines(layout)
 Sobe connectionLineVertices para GPU (VBO).

 RenderFrame(...)
 
 Calcula ViewProj, desenha linhas primeiro (shader de linhas).
 
 Desenha neurônios (esferas) com escala = raio.
 
 Desenha camadas CONV como painéis (folhas) com espessura fina + gap configurável.
 
 Hotkeys (1/2/3/0/Esc) já tratados.
 
 IMPORTANTE: Ajustes comuns (pontos “perguntados com frequência”)
 
 Tamanho das esferas (neurônios)
 Procure por kNeuronRadius em CnnModelLayout::BuildVisualLayersAndComputeBounds().
 A escala da malha da esfera usa diretamente esse raio.
 
 Afastar as folhas da conv (mais espaço em branco)
 Modifique ConvLayerVisual::panelSpacingMultiplier.
 Isso não altera a “grossura” (thickness) da folha, só a distância entre elas.
 
 Largura/altura/profundidade do bloco conv
 São proporcionais a {A,B,C} normalizados pelos máximos do modelo, limitados por:
 
 kWorldMaxWidth, kWorldMaxHeight, kWorldMaxDepth.
 
 Conexões não aparecem?
 Verifique se UploadConnectionLines(layout) é chamado após criar o contexto OpenGL (após CreateWindowAndInitializeGlContext()), e se seu input tem pelo menos 2 estágios.

                                                FIM DO TRECHO SOBRE A MODELAGEM 3D 

# 🧠 Uso do Extrator

O **Extrator** é um conjunto de ferramentas para **monitorar, registrar e visualizar** as ativações e gradientes de redes neurais durante o treinamento.  
Ele permite compreender o comportamento interno de cada camada — convolucional e densa — gerando arquivos JSON detalhados e gráficos que documentam a evolução do modelo.

---

## ⚙️ Demo Executável

Há uma demo executável no arquivo **`demo_train_catsdogs_unified.py`**.  
Essa demo realiza a **extração das informações** durante o treinamento e salva os dados em um arquivo **JSON**.

O JSON resultante é nomeado como: unified_epoch_X_demo_unified_final.json

onde **X** representa o número da época salva.  

Além disso, também são gerados arquivos auxiliares: dense_epoch_X_demo_unified_final.json 

Esses arquivos são temporários e utilizados apenas para compor o JSON unificado.

---

## 🗂️ Estrutura do JSON

### **meta**
Contém informações gerais da execução:
- `epoch`: época à qual os dados se referem  
- `timestamp`: data e hora da execução  
- `run_id`: identificação única da execução  

### **conv**
Dados referentes às camadas convolucionais:
- `epoch`: época correspondente  
- `layers`: lista de camadas convolucionais monitoradas  

Cada camada (ex: `conv1`) contém:
- `H`: altura do Grad-Map  
- `W`: largura do Grad-Map  
- `count`: número de amostras utilizadas para gerar o Grad-Map  
- `map`: tensor com os dados do Grad-Map, utilizado para visualizar os padrões aprendidos pela camada  
- `acts-meta`: metadados das ativações  
  - `H`: altura do mapa de features  
  - `W`: largura do mapa de features  
  - `channels`: número de canais da camada  
  - `imgs`: número de imagens de referência usadas para extrair as ativações  
- `acts`: vetor contendo todas as ativações para as entradas de referência fixadas  

### **dense**
Dados referentes às camadas densas:
- **meta**: informações gerais  
  - `epoch`: época correspondente  
  - `timestamp`: data e hora da execução  
  - `run_id`: identificação da execução  
  - `total_examples`: número de amostras utilizadas para extrair o *Gradient × Activation*  

- **layers**: lista de camadas densas monitoradas  

Cada camada (ex: `fc1`) contém:
- `in_features`: dimensão da entrada  
- `out_features`: dimensão da saída (ou número de neurônios)  
- `ref_inputs`: lista de vetores de entrada da camada para as amostras de referência fixadas  
- `ref_acts`: ativações correspondentes às entradas de referência  
- `heatmap`: vetor contendo o *Gradient × Activation*, que representa a força de cada ligação entre neurônios.  
  Para obter a relevância individual de cada neurônio, deve-se somar e normalizar esses valores.  

### **metrics**
Estatísticas do treinamento até a época atual:
- `acc_per_epoch`: acurácias por época registradas anteriormente
- `acc_last`: acurácia da última época registrada  
- `refs_orig_png_grid`: mosaico contendo todas as imagens de referência utilizadas na rede  
- `ref_indices`: índices do *loader* referentes às imagens fixadas como referência  

---

## 📊 Saídas Adicionais

Além dos arquivos JSON por época, a demo também gera:
- **`acc_plot_epoch_X.png`** → gráfico da evolução da acurácia até a época X.  
  A geração desses gráficos é feita pela função **`_plot_acc_then_png()`**.

- **`ativacao.txt`** → arquivo utilizado para a **visualização 3D da arquitetura da rede**.  
  O código responsável por gerar este arquivo encontra-se em **`gen_architecture.py`**.

- **`refs_orig_i.png`** → i-ésima imagem de referência fixada do loader.

---

## 🖥️ Visualizadores

Três programas adicionais permitem a visualização dos dados contidos nos JSONs:

1. **`visualizador_de_acts_maps_tkinter_matplotlib.py`**  
   → Exibe as ativações das camadas convolucionais como imagens.

2. **`visualizador_de_grad_maps_tkinter_matplotlib.py`**  
   → Exibe os *Layer-Maps* das camadas convolucionais.

3. **`visualizador_de_neuronios_tkinter_matplotlib.py`**  
   → Exibe as ativações e os *Gradient × Activation* das camadas densas.

---

## 🧩 Configuração de uma Nova Rede

Para realizar a extração de dados durante o treinamento com outra rede, basta seguir o exemplo contido em **`demo_train_catsdogs_unified.py`**.  

Caso deseje reproduzir exatamente o procedimento da função `main()`, siga estes passos:
1. Crie uma classe de modelo semelhante à **`SmallCNN`**.  
2. Implemente uma função de *loader* de dataset, semelhante à **`get_catsdogs_loader()`**.

Essas duas partes garantem a compatibilidade com o pipeline de extração e salvamento em JSON.

---
