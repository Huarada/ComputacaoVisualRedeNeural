import os, glob, subprocess, json, re, ast, time, matplotlib as mat

def openNetView(modelo, neuron_up):
    try:
        # Run the compiled C program
        result = subprocess.run(['./neural3d', f'./{modelo}', f'./{neuron_up}'], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        exit()

def model_extract(modelo):
    try:
        with open(modelo, 'r', encoding='utf-8') as f:
            dados_str = f.read()
    except Exception as e:
        print(f"Erro ao ler o arquivo '{modelo}': {e}")
        return []
    
    dados_str = re.sub(r'\{.*?\}', '', dados_str)
    dados_str = dados_str.replace(", ,", "")

    try:
        estrutura_de_dados = ast.literal_eval(dados_str)
    except (ValueError, SyntaxError) as e:
        print(f"Erro de sintaxe após limpeza. String avaliada: {dados_str}")
        print(f"Detalhe do erro: {e}")
        return []

    mod = []
    for lista_interna in estrutura_de_dados:
        if isinstance(lista_interna, list):
            mod.append(len(lista_interna))

    return mod

class Log:
    def __init__(self, model):
        self.Camadas = model
        self.epoca = {}
        self.refNum = 0
    def addRef(self, epoca, camada, ref):
        # Verifica o número bate com a camada
        if (self.Camadas[camada] != len(ref)):
            raise ValueError("Número errado de entradas no log")

        ep = self.epoca.setdefault(epoca, {})
        cam = ep.setdefault(camada, [])
        cam.append(ref)
        self.refNum = len(cam) 
         
def load_logs(log_path, modelo):
    search_pattern = os.path.join(log_path, "*.json")
    log_files = glob.glob(search_pattern)

    total_found = len(log_files)
    if total_found == 0:
        print(f"Nenhum arquivo JSON encontrado na pasta:\n**{log_path}**")
        return None

    successfully_loaded = 0
    logs = Log(modelo)

    # Itera sobre os arquivos e carrega o conteúdo
    for file_path in log_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0] # remove a extensão

        # Carrega apenas das camadas densas e imagens
        if file_name[:11] != "dense_epoch":
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                epoca = data['meta']['epoch']
                layers = data['layers']

                for camada, layer in enumerate(layers):
                    for ref in layers[layer]["ref_acts"]:
                        logs.addRef(epoca, camada, ref)

                successfully_loaded += 1

        except json.JSONDecodeError:
            print(f"Erro: O arquivo '{file_path}' não é um JSON válido.")
        except Exception as e:
            print(f"Erro ao ler o arquivo '{file_path}': {e}")

    if successfully_loaded == 0:
        print(f"Nenhum arquivo JSON pode ser carregado na pasta:\n**{log_path}**")
        return None
    
    return logs

def load_images(image_path, num):
    search_pattern = os.path.join(image_path, "*.png")
    imgs = glob.glob(search_pattern)

    total_found = len(imgs)
    if total_found == 0:
        print(f"Nenhuma imagem encontrado na pasta:\n**{image_path}**")
        return []
    elif total_found < num:
        print(f"Não há imagens suficientes na pasta:\n**{image_path}**")
        return []
    
    return imgs


if __name__ == "__main__":
    model_layout = "ativacoes.txt"
    #model_layout = input("Insira o caminho para o modelo: ").strip()
    modelo = model_extract(model_layout)
    neuron_up = open('neuron_up.txt', "w")
    image_up = open('image_up.txt', "w")
    ref_images = []

    logs = None
    while logs == None:
       # log_path = input("Insira o caminho para a pasta dos json: ").strip()
        log_path = "Extrator/logs_unified_final"
        logs = load_logs(log_path, modelo)

    ref_images = load_images(log_path, logs.refNum)

    p = subprocess.Popen( ['./neural3d', f'./{model_layout}', './neuron_up.txt', './image_up.txt'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    cmap = mat.colormaps.get_cmap("turbo")

    numRef = len(logs.epoca.get(0)[0])
    for (ep, cam) in logs.epoca.items():

        for i in range(0, numRef):
            print(f"Época {ep+1}, imagem = {i+1}")
            neuron_up.seek(0)
            image_up.seek(0)
            image_up.write(ref_images[i])

            for c in cam:
                cam_cores = cmap(cam[c][i])
                for cor in cam_cores:
                    neuron_up.write(f"{cor[0]:.6f} {cor[1]:.6f} {cor[2]:.6f}\n")

            time.sleep(2)
