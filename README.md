# Neural Net 3D Parallel + Conv (OpenGL)

Visualização 3D (OpenGL 3.3) de redes neurais **com estágios em série, branches paralelos** e **camadas convolucionais**.
Lê as ativações de um arquivo `.txt` (ou `stdin`) e renderiza neurônios (esferas), conexões (linhas) e **convs** (blocos “fatiados”).
Cores variam conforme o uso/ativação.

---

## 📦 Requisitos

**SO / Toolchain**
- Windows + MinGW-w64 (recomendado)  
  *(ou WSL/Linux equivalente; ver observação ao final)*

**Bibliotecas**
- **GLFW 3** (headers + import lib + DLL)  
  - Coloque `glfw3.dll` ao lado do executável.
  - Coloque `glfw3dll.a` (ou equivalente) em `./lib/`.
- **GLAD**  
  - O código C já está no projeto em `src/glad.c`.
- Placa com **OpenGL 3.3+**.


---

## 🔨 Compilação (Windows / MinGW)

Dentro da pasta do projeto, rode:

```bash
g++ -std=c++17 -O2 -I include src\glad.c .\neural_net_3d_parallel_conv.cpp ^
  -L lib -lglfw3dll -lopengl32 -lgdi32 -luser32 -lkernel32 -o neural3d.exe

```
-I include → onde estão glad.h e GLFW/glfw3.h

-L lib → onde está glfw3dll.a

-lopengl32 -lgdi32 -luser32 -lkernel32 → libs do Windows/OpenGL

Se já existir neural3d.exe, você pode pular esta etapa.

Dica: se o PowerShell reclamar do ^, use o comando numa linha só.

📝 Formato de entrada (arquivo ou stdin)

O arquivo (ex.: ativacoes.txt) deve conter um array de estágios:

[ stage0 , stage1 , stage2 , ... ]


Cada estágio pode ser:

Denso (sequência de neurônios):

[0, 1, 2]            // três neurônios (contagens 0, 1, 2)


Paralelo (branches): um array contendo sub-arrays de números e/ou convs

[[0,1,2], [1,1], {224x224x64}]


Somente convolucional:

{224x224x64}         // A x B x C


Convolucional {A x B x C}

A×B é a face frontal (plano YZ), desenhada como um conjunto de “fatias”.

C controla a profundidade no eixo X (na direção da próxima camada).

O bloco é mostrado como várias placas finas (3 a 7) para dar sensação de volume.

Observações:

Valores negativos de ativação são convertidos para 0.

É permitido misturar estágios densos e convolucionais, além de paralelismo.

▶️ Executar

Arquivo
```bash
.\neural3d.exe .\ativacoes.txt
```

Via stdin (pipe)
type ativacoes.txt | .\neural3d.exe

Sem fornecer input

Um exemplo default interno será usado:

```bash
[{224x224x64}, [3,2,1], [1]]
```


🎨 Cores (neurônios)
Ativações	Cor
0	cinza
1	azul
2	verde
3	amarelo
4	laranja
5+	vermelho
🖱 Controles

Mouse esquerdo + arrastar → orbitar a câmera ao redor do modelo

Scroll → zoom in/out

Teclas rápidas:

1 → vista frontal

2 → vista lateral direita

3 → vista lateral esquerda

0 → reset (centraliza e ajusta o zoom)

ESC → sair

A câmera orbita em torno do centro do modelo (pivô calculado automaticamente).

🧩 Exemplos de entrada

1) Paralelo simples (duas redes em um estágio):
```bash
[[[0,1],[0,4]]]
```

2) Dois estágios; o segundo com paralelismo:
```bash
[[0,1,2], [[4,2,1],[1,2,4]]]
```

3) Conv seguida de denso e saída:
```bash
[{224x224x64}, [3,2,1], [1]]
```

4) Vários ramos no mesmo estágio (conv + denso):
```bash
[{224x224x64}, {112x112x128}, [1,2,3]]
```
🧠 Como o layout é montado

Eixo X: estágios em série (0, 1, 2, …).

Eixo Z: branches em paralelo dentro de um estágio.

Eixo Y: posição vertical dos neurônios dentro de cada branch.

Convs: blocos com face A×B voltada para a câmera frontal e profundidade C no eixo X (na direção da próxima camada).

Conexões: linhas ligam todos os neurônios/blocos de um estágio aos do estágio seguinte. Para convs, as linhas ancoram nas faces do bloco (sem atravessá-lo).

🚀 Exemplo prático

Crie ativacoes.txt com:
```bash
[[1,2,3], [[0,1],[2,2]], [4,1]]
```

Rode:
```bash
.\neural3d.exe .\ativacoes.txt
```

Você verá:

Estágio 0: 1 → 2 → 3 neurônios.

Estágio 1: paralelo com dois branches ([0,1] e [2,2]).

Estágio 2: 4 → 1 neurônio.

As conexões ligam tudo de um estágio a tudo do seguinte.