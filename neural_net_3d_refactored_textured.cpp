// ============================================================================
// Neural CNN 3D Viewer — Clean Code Edition (single-file)
// ----------------------------------------------------------------------------
//
// Atalhos de câmera: 1 (frontal) / 2 (lado dir) / 3 (lado esq) / 0 (reset)
// ============================================================================

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <queue>
#include <filesystem>
#include <sstream>

// ============================================================================
// Carregador de textura (stb_image)
// ============================================================================
#define STB_IMAGE_IMPLEMENTATION
#include "./include/stb_image.h"

static constexpr float PI = 3.1415926535f;

// ============================================================================
// Matemática utilitária (intencionalmente “verbosa” nos nomes)
// ============================================================================
struct Vec3 { float x, y, z; };

struct Mat4 { float m[16]; };

namespace Math {

static Mat4 MakeIdentityMatrix(){
    Mat4 r{}; r.m[0]=r.m[5]=r.m[10]=r.m[15]=1.0f; return r;
}

static Mat4 MultiplyMat4(const Mat4& a, const Mat4& b){
    Mat4 r{};
    for(int c=0;c<4;++c){
        for(int rI=0;rI<4;++rI){
            r.m[c*4+rI] =
                a.m[0*4+rI]*b.m[c*4+0] +
                a.m[1*4+rI]*b.m[c*4+1] +
                a.m[2*4+rI]*b.m[c*4+2] +
                a.m[3*4+rI]*b.m[c*4+3];
        }
    }
    return r;
}

static Mat4 MakeTranslationMatrix(const Vec3& t){
    Mat4 r = MakeIdentityMatrix();
    r.m[12]=t.x; r.m[13]=t.y; r.m[14]=t.z;
    return r;
}

static Mat4 MakeScaleMatrix(float sx,float sy,float sz){
    Mat4 r{};
    r.m[0]=sx; r.m[5]=sy; r.m[10]=sz; r.m[15]=1.0f;
    return r;
}

static Mat4 MakePerspectiveMatrix(float fovyDeg,float aspect,float n,float f){
    float k = 1.0f / std::tan((fovyDeg*PI/180.0f)*0.5f);
    Mat4 r{};
    r.m[0]=k/aspect; r.m[5]=k; r.m[10]=(f+n)/(n-f); r.m[11]=-1.0f; r.m[14]=(2*f*n)/(n-f);
    return r;
}

static Vec3 NormalizeVec3(const Vec3& v){
    float l = std::sqrt(v.x*v.x+v.y*v.y+v.z*v.z);
    return { v.x/l, v.y/l, v.z/l };
}
static Vec3 CrossVec3(const Vec3& a, const Vec3& b){
    return {
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    };
}
static Mat4 MakeLookAtMatrix(const Vec3& eye,const Vec3& center,const Vec3& up){
    Vec3 f = NormalizeVec3({center.x-eye.x, center.y-eye.y, center.z-eye.z});
    Vec3 s = NormalizeVec3(CrossVec3(f, up));
    Vec3 u = CrossVec3(s, f);
    Mat4 r = MakeIdentityMatrix();
    r.m[0]=s.x; r.m[4]=s.y; r.m[8]=s.z;
    r.m[1]=u.x; r.m[5]=u.y; r.m[9]=u.z;
    r.m[2]=-f.x; r.m[6]=-f.y; r.m[10]=-f.z;
    r.m[12]= -(s.x*eye.x + s.y*eye.y + s.z*eye.z);
    r.m[13]= -(u.x*eye.x + u.y*eye.y + u.z*eye.z);
    r.m[14]=  (f.x*eye.x + f.y*eye.y + f.z*eye.z);
    return r;
}

} // namespace Math

// ============================================================================
// Shaders
// ============================================================================
// Linhas simples
static const char* kLineVS = R"(#version 330 core
layout (location=0) in vec3 aPos;
uniform mat4 uViewProj;
void main(){ gl_Position = uViewProj * vec4(aPos,1.0); }
)";
static const char* kLineFS = R"(#version 330 core
out vec4 FragColor;
uniform vec3 uColor;
void main(){ FragColor = vec4(uColor,1.0); }
)";

// Vertex shader iluminado + textura opcional
static const char* kLitVS = R"(#version 330 core
layout (location=0) in vec3 aPos;
layout (location=1) in vec3 aNormal;
layout (location=2) in vec2 aTexCoord;

uniform mat4 uModel;
uniform mat4 uViewProj;

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vTexCoord;

void main(){
    vec4 wp = uModel * vec4(aPos,1.0);
    vWorldPos = wp.xyz;

    mat3 normalMat = transpose(inverse(mat3(uModel)));
    vNormal = normalize(normalMat * aNormal);

    vTexCoord = aTexCoord;
    gl_Position = uViewProj * wp;
}
)";

// Fragment shader com suporte a textura (usada só na primeira folha da 1ª conv)
static const char* kLitFS = R"(#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vTexCoord;

out vec4 FragColor;

uniform vec3 uBaseColor;
uniform vec3 uLightDir;
uniform vec3 uCamPos;
uniform sampler2D uTexture;
uniform int uUseTexture; // 0 = sem textura, 1 = usa textura na folha selecionada

void main(){
    vec3 n = normalize(vNormal);
    vec3 L = normalize(uLightDir);
    vec3 V = normalize(uCamPos - vWorldPos);

    float diff = max(dot(n,L),0.0);
    float rim  = pow(1.0 - max(dot(n,V),0.0), 2.0);

    vec3 baseColor = uBaseColor;
    if(uUseTexture == 1){
        vec4 texColor = texture(uTexture, vTexCoord);
        baseColor *= texColor.rgb;
    }

    vec3 col = mix(baseColor*0.55, baseColor*1.15, diff) + rim*0.25;
    FragColor = vec4(clamp(col,0.0,1.0),1.0);
}
)";

// ============================================================================
// OpenGL helpers
// ============================================================================
static void OnGlfwError(int code,const char* desc){ std::fprintf(stderr,"GLFW %d: %s\n",code,desc); }

static GLuint CompileShaderOrExit(GLenum shaderType,const char* source){
    GLuint sh = glCreateShader(shaderType);
    glShaderSource(sh,1,&source,nullptr);
    glCompileShader(sh);
    GLint ok=0; glGetShaderiv(sh,GL_COMPILE_STATUS,&ok);
    if(!ok){
        GLint n=0; glGetShaderiv(sh,GL_INFO_LOG_LENGTH,&n);
        std::string log(n,'\0'); glGetShaderInfoLog(sh,n,nullptr,log.data());
        std::fprintf(stderr,"Shader compilation error:\n%s\n",log.c_str());
        std::exit(1);
    }
    return sh;
}
static GLuint LinkProgramOrExit(const char* vs,const char* fs){
    GLuint v=CompileShaderOrExit(GL_VERTEX_SHADER,vs);
    GLuint f=CompileShaderOrExit(GL_FRAGMENT_SHADER,fs);
    GLuint p=glCreateProgram(); glAttachShader(p,v); glAttachShader(p,f); glLinkProgram(p);
    glDeleteShader(v); glDeleteShader(f);
    GLint ok=0; glGetProgramiv(p,GL_LINK_STATUS,&ok);
    if(!ok){
        GLint n=0; glGetProgramiv(p,GL_INFO_LOG_LENGTH,&n);
        std::string log(n,'\0'); glGetProgramInfoLog(p,n,nullptr,log.data());
        std::fprintf(stderr,"Program link error:\n%s\n",log.c_str());
        std::exit(1);
    }
    return p;
}

// Carrega textura 2D de arquivo (image.png na raiz)
static GLuint LoadTexture2DFromFileOrWarn(const char* path){
    int w=0,h=0,ncomp=0;
    stbi_set_flip_vertically_on_load(1);
    unsigned char* data = stbi_load(path, &w, &h, &ncomp, 0);
    if(!data){
        std::fprintf(stderr,"Falha ao carregar textura '%s': %s\n", path, stbi_failure_reason());
        return 0;
    }

    GLenum format = GL_RGB;
    if(ncomp == 1) format = GL_RED;
    else if(ncomp == 3) format = GL_RGB;
    else if(ncomp == 4) format = GL_RGBA;

    GLuint tex=0;
    glGenTextures(1,&tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    glTexImage2D(GL_TEXTURE_2D, 0, format, w, h, 0, format, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);

    stbi_image_free(data);
    glBindTexture(GL_TEXTURE_2D,0);
    return tex;
}

// ============================================================================
// Geometria
// ============================================================================
static void BuildNeuronSphereMesh(int stacks,int slices,
                                  std::vector<float>& vertices,
                                  std::vector<unsigned int>& indices){
    for(int i=0;i<=stacks;++i){
        float vv=float(i)/stacks, phi=vv*PI;
        float y=std::cos(phi), r=std::sin(phi);
        for(int j=0;j<=slices;++j){
            float uu=float(j)/slices, th=uu*2*PI;
            float x=r*std::cos(th), z=r*std::sin(th);
            vertices.insert(vertices.end(),{x,y,z, x,y,z}); // pos + normal
        }
    }
    int stride=slices+1;
    for(int i=0;i<stacks;++i){
        for(int j=0;j<slices;++j){
            unsigned a=i*stride+j, b=(i+1)*stride+j, c=a+1, d=b+1;
            indices.insert(indices.end(),{a,b,c, b,d,c});
        }
    }
}

// Adiciona UVs ao cubo para permitir textura na face frontal
static void BuildConvPanelCubeMesh(std::vector<float>& vertices,
                                   std::vector<unsigned int>& indices){
    struct V{float x,y,z,nx,ny,nz,u,v;};
    std::vector<V> vs={
        // Frente (z = +0.5)  -> UVs completos para textura
        {-0.5f,-0.5f, 0.5f, 0,0,1, 0.0f,0.0f},
        { 0.5f,-0.5f, 0.5f, 0,0,1, 1.0f,0.0f},
        { 0.5f, 0.5f, 0.5f, 0,0,1, 1.0f,1.0f},
        {-0.5f, 0.5f, 0.5f, 0,0,1, 0.0f,1.0f},

        // Traseira (z = -0.5) – UVs genéricos
        {-0.5f,-0.5f,-0.5f, 0,0,-1, 0.0f,0.0f},
        {-0.5f, 0.5f,-0.5f, 0,0,-1, 0.0f,1.0f},
        { 0.5f, 0.5f,-0.5f, 0,0,-1, 1.0f,1.0f},
        { 0.5f,-0.5f,-0.5f, 0,0,-1, 1.0f,0.0f},

        // Esquerda
        {-0.5f,-0.5f,-0.5f,-1,0,0, 0.0f,0.0f},
        {-0.5f,-0.5f, 0.5f,-1,0,0, 1.0f,0.0f},
        {-0.5f, 0.5f, 0.5f,-1,0,0, 1.0f,1.0f},
        {-0.5f, 0.5f,-0.5f,-1,0,0, 0.0f,1.0f},

        // Direita
        { 0.5f,-0.5f,-0.5f, 1,0,0, 0.0f,0.0f},
        { 0.5f, 0.5f,-0.5f, 1,0,0, 0.0f,1.0f},
        { 0.5f, 0.5f, 0.5f, 1,0,0, 1.0f,1.0f},
        { 0.5f,-0.5f, 0.5f, 1,0,0, 1.0f,0.0f},

        // Baixo
        {-0.5f,-0.5f,-0.5f, 0,-1,0, 0.0f,0.0f},
        { 0.5f,-0.5f,-0.5f, 0,-1,0, 1.0f,0.0f},
        { 0.5f,-0.5f, 0.5f, 0,-1,0, 1.0f,1.0f},
        {-0.5f,-0.5f, 0.5f, 0,-1,0, 0.0f,1.0f},

        // Cima
        {-0.5f, 0.5f,-0.5f, 0, 1,0, 0.0f,0.0f},
        {-0.5f, 0.5f, 0.5f, 0, 1,0, 0.0f,1.0f},
        { 0.5f, 0.5f, 0.5f, 0, 1,0, 1.0f,1.0f},
        { 0.5f, 0.5f,-0.5f, 0, 1,0, 1.0f,0.0f}
    };
    unsigned int id[]={
        0,1,2,0,2,3,
        4,5,6,4,6,7,
        8,9,10,8,10,11,
        12,13,14,12,14,15,
        16,17,18,16,18,19,
        20,21,22,20,22,23
    };
    for(auto& p:vs)
        vertices.insert(vertices.end(),{p.x,p.y,p.z,p.nx,p.ny,p.nz,p.u,p.v});
    indices.insert(indices.end(),std::begin(id),std::end(id));
}

// ============================================================================
// Parser do input
// ============================================================================
struct Node{
    enum { NUM, ARR, CONV } kind=NUM;
    long long val=0;
    std::vector<Node> arr;
    int A=0,B=0,C=0; // {AxBxC}
};

static void SkipWhitespace(const std::string& s,size_t& i){
    while(i<s.size() && (s[i]==' '||s[i]=='\n'||s[i]=='\t'||s[i]=='\r'||s[i]==',')) ++i;
}
static bool ParseNumberToken(const std::string& s,size_t& i,long long& out){
    SkipWhitespace(s,i);
    bool neg=false; if(i<s.size()&&s[i]=='-'){neg=true;++i;}
    if(i>=s.size()||!(s[i]>='0'&&s[i]<='9')) return false;
    long long a=0; while(i<s.size()&&(s[i]>='0'&&s[i]<='9')){ a=a*10+(s[i]-'0'); ++i; }
    out=neg?-a:a; return true;
}
static bool ParseConvToken(const std::string& s,size_t& i,int& A,int& B,int& C){
    SkipWhitespace(s,i); if(i>=s.size()||s[i]!='{') return false; ++i;
    long long a=0,b=0,c=0;
    if(!ParseNumberToken(s,i,a)) return false;
    if(i>=s.size()||(s[i]!='x'&&s[i]!='X')) return false; ++i;
    if(!ParseNumberToken(s,i,b)) return false;
    if(i>=s.size()||(s[i]!='x'&&s[i]!='X')) return false; ++i;
    if(!ParseNumberToken(s,i,c)) return false;
    SkipWhitespace(s,i); if(i>=s.size()||s[i]!='}') return false; ++i;
    A=(int)std::max(0LL,a); B=(int)std::max(0LL,b); C=(int)std::max(0LL,c);
    return true;
}
static Node ParseNode(const std::string& s,size_t& i){
    SkipWhitespace(s,i);
    if(i>=s.size()) return {};
    if(s[i]=='{'){ int A,B,C; size_t j=i; if(!ParseConvToken(s,j,A,B,C)) return {}; i=j; Node n; n.kind=Node::CONV; n.A=A;n.B=B;n.C=C; return n; }
    if(s[i]=='['){
        ++i; Node n; n.kind=Node::ARR;
        while(true){
            SkipWhitespace(s,i); if(i>=s.size()) break; if(s[i]==']'){ ++i; break; }
            n.arr.push_back(ParseNode(s,i));
            SkipWhitespace(s,i); if(i<s.size()&&s[i]==']'){ ++i; break; }
            if(i<s.size()&&s[i]==',') ++i;
        }
        return n;
    }
    long long v=0; if(!ParseNumberToken(s,i,v)) return {};
    Node n; n.kind=Node::NUM; n.val=v; return n;
}

// ============================================================================
// Estruturas semânticas
// ============================================================================
struct LayerUnit {
    enum Kind { kNeuron, kConvolution } kind=kNeuron;
    int neuronCount=0;
    int A=0,B=0,C=0;
};

using Branch  = std::vector<LayerUnit>;
using Stage   = std::vector<Branch>;

// ============================================================================
// Visuals
// ============================================================================
class NeuronLayerVisual {
public:
    struct NeuronInstance {
        Vec3 worldCenter;
        float worldRadius;
        float neuronColor[3];
    };
    std::vector<NeuronInstance> instances;

    void AddNeuronInstance(const Vec3& center, float red, float green, float blue, float radius = 0.7f){
        instances.push_back({center, radius, {red, green, blue}});
    }
};

class ConvLayerVisual {
public:
    struct ConvBlockInstance {
        Vec3 worldCenter;
        float worldWidth;
        float worldHeight;
        float worldDepth;
        int A,B,C;
    };
    std::vector<ConvBlockInstance> blocks;

    float baseColorRGB[3] = {0.25f, 0.7f, 1.0f};
    float panelSpacingMultiplier = 7.5f;
};

// ============================================================================
// CnnModelLayout
// ============================================================================
class CnnModelLayout {
public:
    std::vector<Stage> parsedStages;
    std::vector<NeuronLayerVisual> neuronVisuals;
    std::vector<ConvLayerVisual>   convVisuals;

    Vec3  sceneCenter{0,0,0};
    float sceneSpanX=0, sceneSpanY=0, sceneSpanZ=0;

    bool LoadInputAndBuildLayout(int argc,char** argv){
        std::string raw = (argc>1) ? ReadAllFile(argv[1]) : ReadAllStdin();
        if(raw.empty()) raw = "[{224x224x64}, [3,2,1], [1]]";
        size_t i=0; Node root=ParseNode(raw,i);
        parsedStages = ConvertAstToStages(root);
        if(parsedStages.empty()) return false;
        BuildVisualLayersAndComputeBounds();
        return true;
    }
    std::vector<float> connectionLineVertices;

    int numberNeurons = 0;
private:
    struct UnitAnchor {
        Vec3 center;
        bool isConv;
        float halfWidthX;
    };
    std::vector<std::vector<std::vector<UnitAnchor>>> unitAnchors;

    static std::string ReadAllStdin(){
        std::ios::sync_with_stdio(false); std::cin.tie(nullptr);
        std::string s,l; while(std::getline(std::cin,l)){ s+=l; s+=' '; } return s;
    }
    static std::string ReadAllFile(const char* path){
        std::ifstream f(path,std::ios::binary); if(!f) return "";
        return std::string((std::istreambuf_iterator<char>(f)),{});
    }

    static bool NodeIsArrayOfNumbers(const Node& n){
        for(const auto& c:n.arr) if(c.kind!=Node::NUM) return false;
        return true;
    }
    static bool NodeIsArrayOfArraysOrConvs(const Node& n){
        for(const auto& c:n.arr){
            if(c.kind==Node::ARR){ if(!NodeIsArrayOfNumbers(c)) return false; }
            else if(c.kind==Node::CONV) {}
            else return false;
        }
        return true;
    }
    static std::vector<Stage> ConvertAstToStages(const Node& root){
        std::vector<Stage> S; if(root.kind!=Node::ARR) return S;
        for(const auto& s : root.arr){
            Stage stage;
            if(s.kind==Node::CONV){
                Branch b; b.push_back(LayerUnit{LayerUnit::kConvolution,0,s.A,s.B,s.C});
                stage.push_back(std::move(b));
            } else if(s.kind==Node::ARR && NodeIsArrayOfNumbers(s)){
                Branch b;
                for(const auto& e:s.arr) b.push_back(LayerUnit{LayerUnit::kNeuron,(int)std::max(0LL,e.val),0,0,0});
                stage.push_back(std::move(b));
            } else if(s.kind==Node::ARR && NodeIsArrayOfArraysOrConvs(s)){
                for(const auto& it:s.arr){
                    Branch b;
                    if(it.kind==Node::CONV) b.push_back(LayerUnit{LayerUnit::kConvolution,0,it.A,it.B,it.C});
                    else for(const auto& e:it.arr) b.push_back(LayerUnit{LayerUnit::kNeuron,(int)std::max(0LL,e.val),0,0,0});
                    stage.push_back(std::move(b));
                }
            } else {
                Branch b;
                if(s.kind==Node::ARR){
                    for(const auto& e:s.arr){
                        if(e.kind==Node::NUM)  b.push_back(LayerUnit{LayerUnit::kNeuron,(int)std::max(0LL,e.val),0,0,0});
                        if(e.kind==Node::CONV) b.push_back(LayerUnit{LayerUnit::kConvolution,0,e.A,e.B,e.C});
                    }
                }
                if(b.empty()) b.push_back(LayerUnit{LayerUnit::kNeuron,0,0,0,0});
                stage.push_back(std::move(b));
            }
            S.push_back(std::move(stage));
        }
        if(S.empty()){
            S = { Stage{ Branch{ {LayerUnit::kNeuron,0},{LayerUnit::kNeuron,0},{LayerUnit::kNeuron,0},{LayerUnit::kNeuron,0},{LayerUnit::kNeuron,0} } },
                  Stage{ Branch{ {LayerUnit::kNeuron,0},{LayerUnit::kNeuron,0},{LayerUnit::kNeuron,0},{LayerUnit::kNeuron,0} } },
                  Stage{ Branch{ {LayerUnit::kNeuron,0},{LayerUnit::kNeuron,0},{LayerUnit::kNeuron,0} } },
                  Stage{ Branch{ {LayerUnit::kNeuron,0} } } };
        }
        return S;
    }

    void BuildVisualLayersAndComputeBounds(){
        unitAnchors.clear();
        unitAnchors.resize(parsedStages.size());
        neuronVisuals.clear();
        convVisuals.clear();

        int maxA=1,maxB=1,maxC=1;
        for(const auto& st:parsedStages)
            for(const auto& br:st)
                for(const auto& u:br)
                    if(u.kind==LayerUnit::kConvolution){
                        maxA=std::max(maxA,u.A);
                        maxB=std::max(maxB,u.B);
                        maxC=std::max(maxC,u.C);
                    }

        const float kWorldMaxWidth  = 3.4f;
        const float kWorldMaxHeight = 3.4f;
        const float kWorldMaxDepth  = 1.8f;

        const float kNeuronRadius   = 0.7f;
        const float kRowSpacingY    = 2.1f;
        const float kBranchSpacingZ = 2.5f;

        std::vector<float> stageX(parsedStages.size(), 0.0f);

        auto ComputeStageWorldWidth = [&](size_t s) {
            float w = 1.5f;
            for (const auto& br : parsedStages[s]) {
                for (const auto& u : br) {
                    if (u.kind == LayerUnit::kConvolution) {
                        float worldW = std::max(0.6f, kWorldMaxWidth * float(u.A) / float(maxA));
                        w = std::max(w, worldW);
                    }
                }
            }
            return w;
        };
        (void)ComputeStageWorldWidth; // mantido para futura evolução

        // ======== POSICIONAMENTO NO EIXO X ENTRE ESTÁGIOS ========
        stageX[0] = 0.0f;
        for (size_t s = 1; s < parsedStages.size(); ++s) {

            float minSafeGap = 0.0f;

            // largura mínima necessária se o estágio anterior tiver convolução
            for (const auto& br : parsedStages[s-1]) {
                for (const auto& u : br) {
                    if (u.kind == LayerUnit::kConvolution) {
                        float w = std::max(0.6f, kWorldMaxWidth * float(u.A) / float(maxA));
                        minSafeGap = std::max(minSafeGap, w * 0.90f);
                    }
                }
            }

            bool prevHasConv = false;
            for (const auto& br : parsedStages[s-1])
                for (const auto& u : br)
                    if (u.kind == LayerUnit::kConvolution)
                        prevHasConv = true;

            bool currHasConv = false;
            for (const auto& br : parsedStages[s])
                for (const auto& u : br)
                    if (u.kind == LayerUnit::kConvolution)
                        currHasConv = true;

            // gap base padrão
            float gap = 1.4f;

            // ↔ Se for neurônio → neurônio, afastamos mais
            if (!prevHasConv && !currHasConv) {
                gap += 1.2f;   // aumenta a distância entre “colunas” de neurônios
            }

            // leve boost quando o anterior é conv (mantém estética antiga)
            float convBoost = 0.4f;
            float finalGap  = gap + (prevHasConv ? convBoost : 0.0f);

            stageX[s] = stageX[s-1] + std::max(finalGap, minSafeGap);
        }


        float minX= 1e9f, maxX=-1e9f;
        float minY= 1e9f, maxY=-1e9f;
        float minZ= 1e9f, maxZ=-1e9f;

        for(int s=0; s<(int)parsedStages.size(); ++s){
            const auto& stage = parsedStages[s];
            float zStart = -0.5f * (float(stage.size()-1) * kBranchSpacingZ);
            unitAnchors[s].resize(stage.size());

            for(size_t b=0; b<stage.size(); ++b){
                const auto& branch = stage[b];
                float yStart = -0.5f * (float(branch.size()-1) * kRowSpacingY);

                bool branchHasConv=false;
                for(const auto& u : branch) if(u.kind==LayerUnit::kConvolution){ branchHasConv=true; break; }

                if(branchHasConv){
                    ConvLayerVisual convVis;
                    for(size_t i=0;i<branch.size();++i){
                        const auto& u = branch[i]; if(u.kind!=LayerUnit::kConvolution) continue;

                        float worldW = std::max(0.6f, kWorldMaxWidth  * (float)u.A / (float)maxA);
                        float worldH = std::max(0.6f, kWorldMaxHeight * (float)u.B / (float)maxB);
                        float worldD = std::max(0.35f, kWorldMaxDepth * (float)u.C / (float)maxC);

                        Vec3 center { stageX[s],
                                      yStart + i*kRowSpacingY,
                                      zStart + b*kBranchSpacingZ };

                        convVis.blocks.push_back({ center, worldW, worldH, worldD, u.A, u.B, u.C });
                        if (unitAnchors[s][b].size() < branch.size())
                            unitAnchors[s][b].resize(branch.size());
                        unitAnchors[s][b][i] = UnitAnchor{
                            center, true,
                            worldW * 0.5f
                        };

                        float hx = worldW*0.5f, hy=worldH*0.5f, hz=worldD*0.5f;
                        minX = std::min(minX, center.x - hx);  maxX = std::max(maxX, center.x + hx);
                        minY = std::min(minY, center.y - hy);  maxY = std::max(maxY, center.y + hy);
                        minZ = std::min(minZ, center.z - hz);  maxZ = std::max(maxZ, center.z + hz);
                    }
                    convVisuals.push_back(std::move(convVis));
                } else {
                    NeuronLayerVisual neuVis;
                    numberNeurons += branch.size();

                    for(size_t i=0;i<branch.size();++i){
                        const auto& u = branch[i];
                        Vec3 center { stageX[s],
                                      yStart + i*kRowSpacingY,
                                      zStart + b*kBranchSpacingZ };
                        neuVis.AddNeuronInstance(center, 1.0, 1.0, 1.0, kNeuronRadius);
                        if (unitAnchors[s][b].size() < branch.size())
                            unitAnchors[s][b].resize(branch.size());
                        unitAnchors[s][b][i] = UnitAnchor{
                            center, false,
                            kNeuronRadius
                        };

                        float r = kNeuronRadius;
                        minX = std::min(minX, center.x - r);  maxX = std::max(maxX, center.x + r);
                        minY = std::min(minY, center.y - r);  maxY = std::max(maxY, center.y + r);
                        minZ = std::min(minZ, center.z - r);  maxZ = std::max(maxZ, center.z + r);
                    }
                    neuronVisuals.push_back(std::move(neuVis));
                }
            }
        }

        sceneSpanX = maxX - minX; sceneSpanY = maxY - minY; sceneSpanZ = maxZ - minZ;
        sceneCenter = { (minX+maxX)*0.5f, (minY+maxY)*0.5f, (minZ+maxZ)*0.5f };

        connectionLineVertices.clear();
        auto anchorRight = [](const UnitAnchor& u)->Vec3{
            return { u.center.x + u.halfWidthX + 0.02f, u.center.y, u.center.z };
        };
        auto anchorLeft  = [](const UnitAnchor& u)->Vec3{
            return { u.center.x - u.halfWidthX - 0.02f, u.center.y, u.center.z };
        };

        for (size_t s = 0; s + 1 < unitAnchors.size(); ++s) {
            const auto& A = unitAnchors[s];
            const auto& B = unitAnchors[s+1];
            for (const auto& brA : A) for (const auto& uA : brA) {
                Vec3 a = anchorRight(uA);
                for (const auto& brB : B) for (const auto& uB : brB) {
                    Vec3 b = anchorLeft(uB);
                    connectionLineVertices.insert(connectionLineVertices.end(), { a.x,a.y,a.z, b.x,b.y,b.z });
                }
            }
        }
    }
};

// ============================================================================
// SceneRenderer
// ============================================================================
class SceneRenderer {
public:
    GLuint lineProgram=0, litProgram=0;

    GLuint vaoSphere=0, vboSphere=0, iboSphere=0;
    GLuint vaoCube=0,   vboCube=0,   iboCube=0;

    GLint uLinesViewProj=-1, uLinesColor=-1;
    GLint uLitViewProj=-1, uLitModel=-1, uLitBaseColor=-1, uLitLightDir=-1, uLitCamPos=-1;
    GLint uLitUseTexture=-1, uLitTexture=-1;

    // Textura aplicada na primeira folha da primeira camada convolucional
    GLuint convTexture = 0;
    bool   convTextureLoaded = false;

    float orbitalYaw=0.0f, orbitalPitch=0.0f, orbitalRadius=20.0f;
    bool  isRotating=false; double lastMouseX=0, lastMouseY=0;
    Vec3  orbitalTarget{0,0,0};

    Vec3 directionalLight = Math::NormalizeVec3({0.6f,0.7f,0.5f});

    GLsizei sphereIndexCount=0;
    GLsizei cubeIndexCount=0;

    GLuint vaoLines = 0, vboLines = 0;
    GLsizei lineVertexCount = 0;

    std::string lastImage;

    GLFWwindow* CreateWindowAndInitializeGlContext(){
        glfwSetErrorCallback(OnGlfwError);
        if(!glfwInit()){ std::fprintf(stderr,"Falha GLFW\n"); return nullptr; }
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
        glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
        GLFWwindow* win = glfwCreateWindow(1000,700,"Neural CNN 3D — Clean Code",nullptr,nullptr);
        if(!win){ glfwTerminate(); return nullptr; }
        glfwMakeContextCurrent(win); glfwSwapInterval(1);
        if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){ std::fprintf(stderr,"Falha GLAD\n"); return nullptr; }

        glEnable(GL_DEPTH_TEST);
        glClearColor(0.07f,0.08f,0.11f,1.0f);

        lineProgram = LinkProgramOrExit(kLineVS, kLineFS);
        litProgram  = LinkProgramOrExit(kLitVS,  kLitFS);

        uLinesViewProj = glGetUniformLocation(lineProgram,"uViewProj");
        uLinesColor    = glGetUniformLocation(lineProgram,"uColor");

        uLitViewProj   = glGetUniformLocation(litProgram,"uViewProj");
        uLitModel      = glGetUniformLocation(litProgram,"uModel");
        uLitBaseColor  = glGetUniformLocation(litProgram,"uBaseColor");
        uLitLightDir   = glGetUniformLocation(litProgram,"uLightDir");
        uLitCamPos     = glGetUniformLocation(litProgram,"uCamPos");
        uLitUseTexture = glGetUniformLocation(litProgram,"uUseTexture");
        uLitTexture    = glGetUniformLocation(litProgram,"uTexture");

        glUseProgram(litProgram);
        glUniform1i(uLitTexture, 0); // sampler no texture unit 0

        // Esfera dos neurônios (sem UV)
        {
            std::vector<float> verts; std::vector<unsigned int> idx;
            BuildNeuronSphereMesh(24,32,verts,idx);
            sphereIndexCount = (GLsizei)idx.size();

            glGenVertexArrays(1,&vaoSphere); glGenBuffers(1,&vboSphere); glGenBuffers(1,&iboSphere);
            glBindVertexArray(vaoSphere);
            glBindBuffer(GL_ARRAY_BUFFER,vboSphere); glBufferData(GL_ARRAY_BUFFER,verts.size()*sizeof(float),verts.data(),GL_STATIC_DRAW);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,iboSphere); glBufferData(GL_ELEMENT_ARRAY_BUFFER,idx.size()*sizeof(unsigned int),idx.data(),GL_STATIC_DRAW);
            glEnableVertexAttribArray(0); glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,6*sizeof(float),(void*)0);
            glEnableVertexAttribArray(1); glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,6*sizeof(float),(void*)(3*sizeof(float)));
            // atributo 2 (texcoord) permanece desabilitado para a esfera
            glBindVertexArray(0);
        }

        // Cubo/“folha” com UV
        {
            std::vector<float> verts; std::vector<unsigned int> idx;
            BuildConvPanelCubeMesh(verts, idx);
            cubeIndexCount = (GLsizei)idx.size();

            glGenVertexArrays(1,&vaoCube); glGenBuffers(1,&vboCube); glGenBuffers(1,&iboCube);
            glBindVertexArray(vaoCube);
            glBindBuffer(GL_ARRAY_BUFFER,vboCube); glBufferData(GL_ARRAY_BUFFER,verts.size()*sizeof(float),verts.data(),GL_STATIC_DRAW);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,iboCube); glBufferData(GL_ELEMENT_ARRAY_BUFFER,idx.size()*sizeof(unsigned int),idx.data(),GL_STATIC_DRAW);
            glEnableVertexAttribArray(0); glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)0);
            glEnableVertexAttribArray(1); glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)(3*sizeof(float)));
            glEnableVertexAttribArray(2); glVertexAttribPointer(2,2,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)(6*sizeof(float)));
            glBindVertexArray(0);
        }

        // Callback de entrada da câmera
        glfwSetWindowUserPointer(win, this);
        glfwSetMouseButtonCallback(win, [](GLFWwindow* w,int button,int action,int){
            auto* self=(SceneRenderer*)glfwGetWindowUserPointer(w);
            if(button==GLFW_MOUSE_BUTTON_LEFT){
                if(action==GLFW_PRESS){ self->isRotating=true; glfwGetCursorPos(w,&self->lastMouseX,&self->lastMouseY); }
                else if(action==GLFW_RELEASE){ self->isRotating=false; }
            }
        });
        glfwSetCursorPosCallback(win, [](GLFWwindow* w,double x,double y){
            auto* self=(SceneRenderer*)glfwGetWindowUserPointer(w);
            if(!self->isRotating) return;
            double dx=x-self->lastMouseX, dy=y-self->lastMouseY; self->lastMouseX=x; self->lastMouseY=y;
            const float sens=0.005f;
            self->orbitalYaw   += (float)dx*sens;
            self->orbitalPitch += (float)dy*sens;
            float lim=PI/2.0f-0.05f;
            if(self->orbitalPitch> lim) self->orbitalPitch= lim;
            if(self->orbitalPitch<-lim) self->orbitalPitch=-lim;
        });
        glfwSetScrollCallback(win, [](GLFWwindow* w,double, double yoff){
            auto* self=(SceneRenderer*)glfwGetWindowUserPointer(w);
            if(yoff>0) self->orbitalRadius*=0.9f; else self->orbitalRadius*=1.1f;
            self->orbitalRadius = std::clamp(self->orbitalRadius, 5.0f, 300.0f);
        });

        // Carrega textura da primeira folha da primeira camada convolucional
        // convTexture = LoadTexture2DFromFileOrWarn("image.png");
        // convTextureLoaded = (convTexture != 0);

        return win;
    }

    bool UpdateTexture(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            // Arquivo não existe ou não pode ser aberto
            return false;
        }

        std::string texPath;
        if (!std::getline(file, texPath)) {
            // Arquivo vazio
            return false;
        }

        if (texPath == lastImage) {
            return false;
        }

        lastImage = texPath;

        // Remove espaços em branco no início e fim
        texPath.erase(0, texPath.find_first_not_of(" \t\r\n"));
        texPath.erase(texPath.find_last_not_of(" \t\r\n") + 1);

        if (texPath.empty()) {
            // Linha inválida
            return false;
        }

        // Destroi a textura anterior, se existir
        if (convTexture != 0) {
            glDeleteTextures(1, &convTexture);
            convTexture = 0;
        }

        // Carrega textura nova
        GLuint newTex = LoadTexture2DFromFileOrWarn(texPath.c_str());
        if (newTex == 0) {
            return false; // Erro ao carregar textura
        }

        convTexture = newTex;
        convTextureLoaded = 1;
        return true;
    }

    // Verifica se houve atualização nas cores dos neurônios
    bool TryLoadColors(const std::string& filepath, CnnModelLayout& layout) {
        std::ifstream file(filepath);
        if (!file)
            // Arquivo não existe ou não pode ser aberto (não é um erro)
            return false; 

        // Prepara para carregar novas cores de neurônios
        std::queue<float*> loadedNeuronColors;
        //bool inNeuronColorBlock = false;
        std::string line;

        while (std::getline(file, line)) {
            line.erase(0, line.find_first_not_of(" \t\r\n"));

            if (line.empty()) {
                continue;
            }

            std::stringstream ss(line);
            float* color = new float[3];
            ss >> color[0] >> color[1] >> color[2];

            loadedNeuronColors.push(color);
        }

        if (loadedNeuronColors.size() < layout.numberNeurons) // Não há neurônios suficientes 
            return false;

        // Atualiza as cores
        for(auto& layer : layout.neuronVisuals){
            for (auto& n : layer.instances){
                n.neuronColor[0] = loadedNeuronColors.front()[0];
                n.neuronColor[1] = loadedNeuronColors.front()[1];
                n.neuronColor[2] = loadedNeuronColors.front()[2];
                loadedNeuronColors.pop();
            }
        }

        return true;
    }

    void UploadConnectionLines(const CnnModelLayout& layout){
        if (vaoLines) { glDeleteVertexArrays(1,&vaoLines); vaoLines=0; }
        if (vboLines) { glDeleteBuffers(1,&vboLines); vboLines=0; }

        lineVertexCount = (GLsizei)(layout.connectionLineVertices.size() / 3);
        if (lineVertexCount == 0) return;

        glGenVertexArrays(1,&vaoLines);
        glGenBuffers(1,&vboLines);
        glBindVertexArray(vaoLines);
        glBindBuffer(GL_ARRAY_BUFFER, vboLines);
        glBufferData(GL_ARRAY_BUFFER,
                     layout.connectionLineVertices.size()*sizeof(float),
                     layout.connectionLineVertices.data(),
                     GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,3*sizeof(float),(void*)0);
        glBindVertexArray(0);
    }

    void SetCameraTargetAndFitScene(const Vec3& center, float spanX, float spanY, float spanZ){
        orbitalTarget = center;
        orbitalRadius = std::max(12.0f, std::max({spanX,spanY,spanZ})*1.6f);
    }

    void RenderFrame(GLFWwindow* win, const CnnModelLayout& layout){
        glfwPollEvents();
        HandleQuickViewHotkeys(win, layout);

        int fbw,fbh; glfwGetFramebufferSize(win,&fbw,&fbh);
        glViewport(0,0,fbw,fbh);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

        float cp=std::cos(orbitalPitch), sp=std::sin(orbitalPitch), cy=std::cos(orbitalYaw), sy=std::sin(orbitalYaw);
        Vec3 cam{ orbitalTarget.x + orbitalRadius*cp*sy,
                  orbitalTarget.y + orbitalRadius*sp,
                  orbitalTarget.z + orbitalRadius*cp*cy };

        Mat4 proj = Math::MakePerspectiveMatrix(45.0f,(float)fbw/(float)fbh,0.1f,300.0f);
        Mat4 view = Math::MakeLookAtMatrix(cam, orbitalTarget, {0,1,0});
        Mat4 vp   = Math::MultiplyMat4(proj, view);

        // Linhas
        if (lineVertexCount > 0) {
            glUseProgram(lineProgram);
            glUniformMatrix4fv(uLinesViewProj, 1, GL_FALSE, vp.m);
            glUniform3f(uLinesColor, 0.75f, 0.75f, 0.8f);
            glBindVertexArray(vaoLines);
            glLineWidth(1.2f);
            glDrawArrays(GL_LINES, 0, lineVertexCount);
            glBindVertexArray(0);
        }

        // Programa iluminado
        glUseProgram(litProgram);
        glUniformMatrix4fv(uLitViewProj,1,GL_FALSE,vp.m);
        glUniform3f(uLitLightDir,directionalLight.x,directionalLight.y,directionalLight.z);
        glUniform3f(uLitCamPos,cam.x,cam.y,cam.z);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindTexture(GL_TEXTURE_2D, convTexture);
        DrawNeuronLayerVisuals(layout.neuronVisuals);
        DrawConvolutionLayerVisuals(layout.convVisuals);

        glfwSwapBuffers(win);
    }

    void DestroyGlResources(){
        if(vaoSphere){ glDeleteVertexArrays(1,&vaoSphere); vaoSphere=0; }
        if(vboSphere){ glDeleteBuffers(1,&vboSphere); vboSphere=0; }
        if(iboSphere){ glDeleteBuffers(1,&iboSphere); iboSphere=0; }

        if(vaoCube){ glDeleteVertexArrays(1,&vaoCube); vaoCube=0; }
        if(vboCube){ glDeleteBuffers(1,&vboCube); vboCube=0; }
        if(iboCube){ glDeleteBuffers(1,&iboCube); iboCube=0; }

        if(lineProgram){ glDeleteProgram(lineProgram); lineProgram=0; }
        if(litProgram){ glDeleteProgram(litProgram); litProgram=0; }
        if (vaoLines){ glDeleteVertexArrays(1,&vaoLines); vaoLines=0; }
        if (vboLines){ glDeleteBuffers(1,&vboLines); vboLines=0; }
        if (convTexture){ glDeleteTextures(1,&convTexture); convTexture=0; }
    }

private:
    void DrawNeuronLayerVisuals(const std::vector<NeuronLayerVisual>& visuals){
        glBindVertexArray(vaoSphere);
        for(const auto& layer : visuals){
            for(const auto& n : layer.instances){
                Mat4 M = Math::MultiplyMat4(
                    Math::MakeTranslationMatrix(n.worldCenter),
                    Math::MakeScaleMatrix(n.worldRadius, n.worldRadius, n.worldRadius)
                );
                glUniformMatrix4fv(uLitModel,1,GL_FALSE,M.m);
                glUniform3f(uLitBaseColor,n.neuronColor[0], n.neuronColor[1], n.neuronColor[2]);
                glUniform1i(uLitUseTexture, 0); // neurônios não usam textura
                glDrawElements(GL_TRIANGLES, sphereIndexCount, GL_UNSIGNED_INT, 0);
            }
        }
        glBindVertexArray(0);
    }

    // Aqui aplicamos a textura SOMENTE na folha final da cadeia convolucional
        // Aplica a textura SOMENTE na "folha final" da PRIMEIRA camada convolucional
    void DrawConvolutionLayerVisuals(const std::vector<ConvLayerVisual>& visuals){
        glBindVertexArray(vaoCube);

        if (visuals.empty()) {
            glBindVertexArray(0);
            return;
        }

        // --------------------------------------------------------------------
        // 1) Descobrir o alvo: primeira camada convolucional do pipeline
        //    → assumimos visuals[0] como "primeira conv"
        //    → e o primeiro bloco dessa camada como "primeiro bloco conv"
        //    → queremos a ÚLTIMA folha (k = C-1) desse bloco.
        // --------------------------------------------------------------------
        const ConvLayerVisual& firstConvLayer = visuals.front();
        if (firstConvLayer.blocks.empty()) {
            glBindVertexArray(0);
            return;
        }

        const auto& firstBlock = firstConvLayer.blocks.front();
        const int panelsInFirstBlock      = std::max(1, firstBlock.C);
        const int targetPanelGlobalIndex  = panelsInFirstBlock - 1; // folha final da 1ª conv

        // --------------------------------------------------------------------
        // 2) Render: iteramos por TODAS as folhas, mas só aplicamos textura
        //    quando currentPanelIndex == targetPanelGlobalIndex.
        // --------------------------------------------------------------------
        int currentPanelIndex = 0;

        for (const auto& layer : visuals) {
            for (const auto& block : layer.blocks) {
                const int totalPanels = std::max(1, block.C);

                const float panelThickness = (block.worldDepth / float(totalPanels)) * 0.2f;
                const float panelGapFactor = layer.panelSpacingMultiplier;

                const float zStart = -0.5f * block.worldDepth;

                for (int k = 0; k < totalPanels; ++k, ++currentPanelIndex) {
                    const float zOffset = zStart + k * (panelThickness * panelGapFactor);

                    const float t = 0.65f + 0.35f * (float)k / (float)totalPanels;
                    float r = layer.baseColorRGB[0] * t;
                    float g = layer.baseColorRGB[1] * t;
                    float b = layer.baseColorRGB[2] * t;

                    bool useTextureHere =
                        (convTextureLoaded && currentPanelIndex == targetPanelGlobalIndex);

                    if (useTextureHere) {
                        // base branca pra textura mandar na cor final
                        r = g = b = 1.0f;
                    }

                    Mat4 model = Math::MultiplyMat4(
                        Math::MakeTranslationMatrix({
                            block.worldCenter.x,
                            block.worldCenter.y,
                            block.worldCenter.z + zOffset
                        }),
                        Math::MakeScaleMatrix(block.worldWidth, block.worldHeight, panelThickness)
                    );

                    glUniformMatrix4fv(uLitModel, 1, GL_FALSE, model.m);
                    glUniform3f(uLitBaseColor, r, g, b);
                    glUniform1i(uLitUseTexture, useTextureHere ? 1 : 0);

                    glDrawElements(GL_TRIANGLES, cubeIndexCount, GL_UNSIGNED_INT, 0);
                }
            }
        }

        glBindVertexArray(0);
    }

    void HandleQuickViewHotkeys(GLFWwindow* win, const CnnModelLayout& layout){
        if(glfwGetKey(win,GLFW_KEY_1)==GLFW_PRESS){ orbitalYaw=0; orbitalPitch=0; }
        if(glfwGetKey(win,GLFW_KEY_2)==GLFW_PRESS){ orbitalYaw= PI/2; orbitalPitch=0; }
        if(glfwGetKey(win,GLFW_KEY_3)==GLFW_PRESS){ orbitalYaw=-PI/2; orbitalPitch=0; }
        if(glfwGetKey(win,GLFW_KEY_0)==GLFW_PRESS){
            orbitalYaw=0; orbitalPitch=0;
            SetCameraTargetAndFitScene(layout.sceneCenter, layout.sceneSpanX, layout.sceneSpanY, layout.sceneSpanZ);
        }
        if(glfwGetKey(win,GLFW_KEY_ESCAPE)==GLFW_PRESS) glfwSetWindowShouldClose(win,GL_TRUE);
    }
};

// ============================================================================
// main
// ============================================================================
int main(int argc,char** argv){
    CnnModelLayout layout;
    if(!layout.LoadInputAndBuildLayout(argc, argv)){
        std::fprintf(stderr,"Falha ao carregar modelo (input vazio ou inválido)\n");
        return 1;
    }

    SceneRenderer renderer;
    GLFWwindow* win = renderer.CreateWindowAndInitializeGlContext();
    renderer.UploadConnectionLines(layout);

    if(!win) return 1;

    renderer.SetCameraTargetAndFitScene(layout.sceneCenter, layout.sceneSpanX, layout.sceneSpanY, layout.sceneSpanZ);

    while(!glfwWindowShouldClose(win)){
        if (argc > 2) {
            renderer.TryLoadColors(argv[2], layout);
        }
        if (argc > 3) {
            renderer.UpdateTexture(argv[3]);
        }

        renderer.RenderFrame(win, layout);
    }

    renderer.DestroyGlResources();
    glfwDestroyWindow(win); glfwTerminate();
    return 0;
}