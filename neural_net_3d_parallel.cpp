// neural_net_3d_parallel.cpp
// Compilar (Windows MinGW):
// g++ -std=c++17 -O2 -I include src\glad.c neural_net_3d_parallel.cpp -L lib -lglfw3dll -lopengl32 -lgdi32 -luser32 -lkernel32 -o neural3d.exe

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

static constexpr float PI = 3.1415926535f;

struct Vec3 { float x, y, z; };

// ===== Shaders (Core 3.3) =====
static const char* LINE_VS = R"(#version 330 core
layout (location=0) in vec3 aPos;
uniform mat4 uViewProj;
void main(){ gl_Position = uViewProj * vec4(aPos,1.0); }
)";
static const char* LINE_FS = R"(#version 330 core
out vec4 FragColor;
uniform vec3 uColor;
void main(){ FragColor = vec4(uColor,1.0); }
)";
static const char* SPHERE_VS = R"(#version 330 core
layout (location=0) in vec3 aPos;
layout (location=1) in vec3 aNormal;

uniform mat4 uModel;
uniform mat4 uViewProj;

out vec3 vWorldPos;
out vec3 vNormal;

void main(){
    vec4 worldPos = uModel * vec4(aPos,1.0);
    vWorldPos = worldPos.xyz;
    mat3 N = mat3(uModel); // escala uniforme
    vNormal = normalize(N * aNormal);
    gl_Position = uViewProj * worldPos;
}
)";
static const char* SPHERE_FS = R"(#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;
out vec4 FragColor;

uniform vec3 uBaseColor;
uniform vec3 uLightDir; // normalizado
uniform vec3 uCamPos;

void main(){
    vec3 n = normalize(vNormal);
    vec3 L = normalize(uLightDir);
    vec3 V = normalize(uCamPos - vWorldPos);

    float diff = max(dot(n, L), 0.0);
    float rim  = pow(1.0 - max(dot(n, V), 0.0), 2.0);

    vec3 baseDark  = uBaseColor * 0.55;
    vec3 baseLight = uBaseColor * 1.15;
    vec3 color = mix(baseDark, baseLight, diff);
    color += rim * 0.25;
    color = clamp(color, 0.0, 1.0);

    FragColor = vec4(color, 1.0);
}
)";

// ===== GL helpers =====
static void glfwErrorCb(int code, const char* desc){ std::fprintf(stderr,"GLFW error %d: %s\n",code,desc); }
static GLuint compile(GLenum type, const char* src){
    GLuint s = glCreateShader(type); glShaderSource(s,1,&src,nullptr); glCompileShader(s);
    GLint ok=0; glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
    if(!ok){ GLint len=0; glGetShaderiv(s,GL_INFO_LOG_LENGTH,&len); std::string log(len,'\0'); glGetShaderInfoLog(s,len,nullptr,log.data()); std::fprintf(stderr,"Shader compile error: %s\n",log.c_str()); std::exit(EXIT_FAILURE); }
    return s;
}
static GLuint linkProgram(const char* vs, const char* fs){
    GLuint v=compile(GL_VERTEX_SHADER,vs), f=compile(GL_FRAGMENT_SHADER,fs);
    GLuint p=glCreateProgram(); glAttachShader(p,v); glAttachShader(p,f); glLinkProgram(p); glDeleteShader(v); glDeleteShader(f);
    GLint ok=0; glGetProgramiv(p,GL_LINK_STATUS,&ok);
    if(!ok){ GLint len=0; glGetProgramiv(p,GL_INFO_LOG_LENGTH,&len); std::string log(len,'\0'); glGetProgramInfoLog(p,len,nullptr,log.data()); std::fprintf(stderr,"Program link error: %s\n",log.c_str()); std::exit(EXIT_FAILURE); }
    return p;
}

// ===== Mat4 minimal =====
struct Mat4{ float m[16]; };
static Mat4 matIdentity(){ Mat4 r{}; r.m[0]=r.m[5]=r.m[10]=r.m[15]=1; return r; }
static Mat4 matMul(const Mat4& a,const Mat4& b){
    Mat4 r{};
    for(int c=0;c<4;++c) for(int rI=0;rI<4;++rI)
        r.m[c*4+rI]=a.m[0*4+rI]*b.m[c*4+0]+a.m[1*4+rI]*b.m[c*4+1]+a.m[2*4+rI]*b.m[c*4+2]+a.m[3*4+rI]*b.m[c*4+3];
    return r;
}
static Mat4 matTranslate(const Vec3& t){ Mat4 r=matIdentity(); r.m[12]=t.x; r.m[13]=t.y; r.m[14]=t.z; return r; }
static Mat4 matScale(float s){ Mat4 r{}; r.m[0]=r.m[5]=r.m[10]=s; r.m[15]=1; return r; }
static Mat4 matPerspective(float fovyDeg,float aspect,float n,float f){
    float k=1.0f/std::tan(fovyDeg*PI/180.0f/2.0f);
    Mat4 r{}; r.m[0]=k/aspect; r.m[5]=k; r.m[10]=(f+n)/(n-f); r.m[11]=-1; r.m[14]=(2*f*n)/(n-f); return r;
}
static Vec3 normalize(const Vec3& v){ float l=std::sqrt(v.x*v.x+v.y*v.y+v.z*v.z); return {v.x/l,v.y/l,v.z/l}; }
static Vec3 cross(const Vec3& a,const Vec3& b){ return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }
static float dotv(const Vec3& a,const Vec3& b){ return a.x*b.x+a.y*b.y+a.z*b.z; }
static Mat4 matLookAt(const Vec3& eye,const Vec3& center,const Vec3& up){
    Vec3 f=normalize({center.x-eye.x,center.y-eye.y,center.z-eye.z});
    Vec3 s=normalize(cross(f,up)); Vec3 u=cross(s,f);
    Mat4 r=matIdentity();
    r.m[0]=s.x; r.m[4]=s.y; r.m[8]=s.z;
    r.m[1]=u.x; r.m[5]=u.y; r.m[9]=u.z;
    r.m[2]=-f.x; r.m[6]=-f.y; r.m[10]=-f.z;
    r.m[12]=-dotv(s,eye); r.m[13]=-dotv(u,eye); r.m[14]= dotv(f,eye);
    return r;
}

// ===== leitura =====
static std::string readAllStdin(){ std::ios::sync_with_stdio(false); std::cin.tie(nullptr); std::string s,l; while(std::getline(std::cin,l)){ s+=l; s+=' '; } return s; }
static std::string readAllFile(const char* path){ std::ifstream f(path, std::ios::binary); if(!f) return ""; return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()); }

// ===== Parser genérico para arrays de arrays de números =====
struct Node{ bool isNum=false; long long val=0; std::vector<Node> arr; };
static void skipWS(const std::string& s, size_t& i){ while(i<s.size() && (s[i]==' '||s[i]=='\n'||s[i]=='\t'||s[i]=='\r'||s[i]==',')) ++i; }
static Node parseNode(const std::string& s, size_t& i){
    skipWS(s,i);
    if(i>=s.size()) return {};
    if(s[i]=='['){
        ++i; Node n; n.isNum=false;
        while(true){
            skipWS(s,i);
            if(i>=s.size()) break;
            if(s[i]==']'){ ++i; break; }
            n.arr.push_back(parseNode(s,i));
            skipWS(s,i);
            if(i<s.size() && s[i]==']'){ ++i; break; }
            if(i<s.size() && s[i]==','){ ++i; }
        }
        return n;
    } else {
        // número
        int sign=1; if(s[i]=='-'){ sign=-1; ++i; }
        long long acc=0; while(i<s.size() && s[i]>='0' && s[i]<='9'){ acc=acc*10 + (s[i]-'0'); ++i; }
        Node n; n.isNum=true; n.val=sign*acc; return n;
    }
}
// transforma para estágios paralelos: vector<Stage>, Stage = vector<branch>, branch = vector<int>
using Stage = std::vector<std::vector<int>>;
static bool allNums(const Node& n){ for(const auto& c:n.arr) if(!c.isNum) return false; return true; }
static bool allArraysOfNums(const Node& n){ for(const auto& c:n.arr){ if(c.isNum) return false; if(!allNums(c)) return false; } return true; }

static std::vector<Stage> toStages(const Node& root){
    std::vector<Stage> stages;
    if(root.isNum) return stages;
    for(const auto& stageNode : root.arr){
        if(stageNode.isNum) continue;
        Stage stage;
        if(allNums(stageNode)){
            // estágio com 1 branch
            std::vector<int> branch;
            branch.reserve(stageNode.arr.size());
            for(const auto& numNode : stageNode.arr) branch.push_back((int)std::max(0LL, numNode.val));
            stage.push_back(branch);
        } else if(allArraysOfNums(stageNode)){
            // estágio com N branches
            for(const auto& branchNode : stageNode.arr){
                std::vector<int> branch;
                branch.reserve(branchNode.arr.size());
                for(const auto& numNode : branchNode.arr) branch.push_back((int)std::max(0LL, numNode.val));
                stage.push_back(branch);
            }
        } else {
            // degrau mais profundo: achata um nível de forma conservadora
            // pega todos números encontrados no primeiro nível interno
            std::vector<int> branch;
            for(const auto& c: stageNode.arr){
                if(c.isNum) branch.push_back((int)std::max(0LL, c.val));
                else for(const auto& d: c.arr) if(d.isNum) branch.push_back((int)std::max(0LL, d.val));
            }
            if(branch.empty()) branch.push_back(0);
            stage.push_back(branch);
        }
        stages.push_back(stage);
    }
    if(stages.empty()) stages.push_back(Stage{std::vector<int>{0}});
    return stages;
}

// ===== cores =====
static void colorForCount(int c, float outRGB[3]){
    if(c<=0){ outRGB[0]=0.85f; outRGB[1]=0.85f; outRGB[2]=0.85f; return; } // 0 cinza
    if(c==1){ outRGB[0]=0.20f; outRGB[1]=0.40f; outRGB[2]=1.00f; return; } // 1 azul
    if(c==2){ outRGB[0]=0.20f; outRGB[1]=1.00f; outRGB[2]=0.20f; return; } // 2 verde
    if(c==3){ outRGB[0]=1.00f; outRGB[1]=0.95f; outRGB[2]=0.10f; return; } // 3 amarelo
    if(c==4){ outRGB[0]=1.00f; outRGB[1]=0.60f; outRGB[2]=0.10f; return; } // 4 laranja
    outRGB[0]=1.00f; outRGB[1]=0.25f; outRGB[2]=0.25f;                      // 5+ vermelho
}

// ===== esfera =====
static void buildSphereMesh(int stacks, int slices, std::vector<float>& verts, std::vector<unsigned int>& idx){
    for(int i=0;i<=stacks;++i){
        float v=(float)i/(float)stacks, phi=v*PI;
        float y=std::cos(phi), r=std::sin(phi);
        for(int j=0;j<=slices;++j){
            float u=(float)j/(float)slices, th=u*2*PI;
            float x=r*std::cos(th), z=r*std::sin(th);
            verts.push_back(x); verts.push_back(y); verts.push_back(z);
            verts.push_back(x); verts.push_back(y); verts.push_back(z);
        }
    }
    int stride=slices+1;
    for(int i=0;i<stacks;++i){
        for(int j=0;j<slices;++j){
            unsigned a=i*stride+j, b=(i+1)*stride+j, c=a+1, d=b+1;
            idx.push_back(a); idx.push_back(b); idx.push_back(c);
            idx.push_back(b); idx.push_back(d); idx.push_back(c);
        }
    }
}

// ===== posições com paralelismo por estágio =====
struct NeuronPos { Vec3 p; int stageIdx; int branchIdx; int neuronIdx; };
static std::vector<std::vector<std::vector<Vec3>>> computePositions(const std::vector<Stage>& stages){
    int L = (int)stages.size();
    std::vector<std::vector<std::vector<Vec3>>> pos(L); // [stage][branch][neuron]
    float x0=-6.0f, x1=6.0f, dx = (L>1)? (x1-x0)/(L-1) : 0.0f;

    for(int s=0;s<L;++s){
        int B = (int)stages[s].size();
        pos[s].resize(B);
        float zSpacing = 2.2f;
        float zStart = -0.5f * (B-1) * zSpacing;
        for(int b=0;b<B;++b){
            int N = (int)stages[s][b].size();
            pos[s][b].resize(N);
            float ySpacing = 1.8f;
            float yStart = -0.5f * (N-1) * ySpacing;
            for(int i=0;i<N;++i){
                pos[s][b][i] = { x0 + s*dx, yStart + i*ySpacing, zStart + b*zSpacing };
            }
        }
    }
    return pos;
}

// ===== câmera orbit =====
static float gYaw=0.0f, gPitch=0.0f, gRadius=20.0f;
static bool gRot=false; static double gLastX=0.0, gLastY=0.0;
static void mouseButtonCallback(GLFWwindow* w,int button,int action,int mods){
    if(button==GLFW_MOUSE_BUTTON_LEFT){ if(action==GLFW_PRESS){ gRot=true; glfwGetCursorPos(w,&gLastX,&gLastY);} else if(action==GLFW_RELEASE){ gRot=false; } }
}
static void cursorPosCallback(GLFWwindow* w,double x,double y){
    if(!gRot) return;
    double dx=x-gLastX, dy=y-gLastY; gLastX=x; gLastY=y;
    const float sens=0.005f;
    gYaw   += (float)dx*sens;
    gPitch += (float)dy*sens;
    float lim = PI/2.0f - 0.05f;
    if(gPitch> lim) gPitch= lim;
    if(gPitch<-lim) gPitch=-lim;
}

int main(int argc,char** argv){
    // lê string
    std::string raw = (argc>1)? readAllFile(argv[1]) : readAllStdin();
    if(raw.empty()) raw = "[[0,1,2], [[4,2,1],[1,2,4]]]";

    // parse
    size_t i=0; Node root = parseNode(raw, i);
    auto stages = toStages(root);
    int NUM_STAGES = (int)stages.size();

    // GL init
    glfwSetErrorCallback(glfwErrorCb);
    if(!glfwInit()){ std::fprintf(stderr,"Falha GLFW\n"); return EXIT_FAILURE; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* win = glfwCreateWindow(1000,700,"Neural Net 3D - Paralelo",nullptr,nullptr);
    if(!win){ std::fprintf(stderr,"Falha janela\n"); glfwTerminate(); return EXIT_FAILURE; }
    glfwMakeContextCurrent(win); glfwSwapInterval(1);
    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){ std::fprintf(stderr,"Falha GLAD\n"); return EXIT_FAILURE; }

    glfwSetMouseButtonCallback(win, mouseButtonCallback);
    glfwSetCursorPosCallback(win, cursorPosCallback);

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.07f,0.08f,0.11f,1.0f);

    GLuint progLines  = linkProgram(LINE_VS, LINE_FS);
    GLuint progSphere = linkProgram(SPHERE_VS, SPHERE_FS);

    // luz
    Vec3 lightDir = normalize(Vec3{0.6f,0.7f,0.5f});
    const Vec3 target{0,0,0};

    // esfera compartilhada
    std::vector<float> sphereVerts; std::vector<unsigned int> sphereIdx;
    buildSphereMesh(24,32,sphereVerts,sphereIdx);
    GLuint vaoSphere=0,vboSphere=0,eboSphere=0;
    glGenVertexArrays(1,&vaoSphere);
    glGenBuffers(1,&vboSphere);
    glGenBuffers(1,&eboSphere);
    glBindVertexArray(vaoSphere);
    glBindBuffer(GL_ARRAY_BUFFER, vboSphere);
    glBufferData(GL_ARRAY_BUFFER, sphereVerts.size()*sizeof(float), sphereVerts.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eboSphere);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphereIdx.size()*sizeof(unsigned int), sphereIdx.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,6*sizeof(float),(void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,6*sizeof(float),(void*)(3*sizeof(float)));
    glBindVertexArray(0);

    // posições
    auto pos = computePositions(stages);

    // linhas: liga todos do estágio s a todos do estágio s+1
    std::vector<float> lineVerts;
    for(int s=0;s<NUM_STAGES-1;++s){
        for(size_t b0=0;b0<stages[s].size();++b0)
            for(size_t i0=0;i0<stages[s][b0].size();++i0)
                for(size_t b1=0;b1<stages[s+1].size();++b1)
                    for(size_t i1=0;i1<stages[s+1][b1].size();++i1){
                        Vec3 a = pos[s][b0][i0];
                        Vec3 b = pos[s+1][b1][i1];
                        lineVerts.insert(lineVerts.end(), {a.x,a.y,a.z, b.x,b.y,b.z});
                    }
    }
    GLuint vaoLines=0,vboLines=0;
    glGenVertexArrays(1,&vaoLines);
    glGenBuffers(1,&vboLines);
    glBindVertexArray(vaoLines);
    glBindBuffer(GL_ARRAY_BUFFER, vboLines);
    glBufferData(GL_ARRAY_BUFFER, lineVerts.size()*sizeof(float), lineVerts.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,3*sizeof(float),(void*)0);
    glBindVertexArray(0);

    // uniforms
    GLint uVP_L = glGetUniformLocation(progLines,  "uViewProj");
    GLint uColL = glGetUniformLocation(progLines,  "uColor");
    GLint uVP_S   = glGetUniformLocation(progSphere,"uViewProj");
    GLint uModelS = glGetUniformLocation(progSphere,"uModel");
    GLint uBase   = glGetUniformLocation(progSphere,"uBaseColor");
    GLint uLight  = glGetUniformLocation(progSphere,"uLightDir");
    GLint uCam    = glGetUniformLocation(progSphere,"uCamPos");

    // ajusta raio com base no tamanho do grafo
    float maxX= -1e9f, minX= 1e9f, maxY=-1e9f, minY=1e9f, maxZ=-1e9f, minZ=1e9f;
    for(size_t s=0;s<pos.size();++s) for(size_t b=0;b<pos[s].size();++b) for(size_t n=0;n<pos[s][b].size();++n){
        Vec3 p=pos[s][b][n];
        maxX=std::max(maxX,p.x); minX=std::min(minX,p.x);
        maxY=std::max(maxY,p.y); minY=std::min(minY,p.y);
        maxZ=std::max(maxZ,p.z); minZ=std::min(minZ,p.z);
    }
    float span = std::max({maxX-minX, maxY-minY, maxZ-minZ});
    gRadius = std::max(12.0f, span*1.6f);

    const float radiusSphere = 0.7f;

    while(!glfwWindowShouldClose(win)){
        glfwPollEvents();
        if(glfwGetKey(win, GLFW_KEY_ESCAPE)==GLFW_PRESS) glfwSetWindowShouldClose(win, GLFW_TRUE);

        int fbw, fbh; glfwGetFramebufferSize(win,&fbw,&fbh);
        glViewport(0,0,fbw,fbh);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // câmera orbit em torno do centro
        float cp=std::cos(gPitch), sp=std::sin(gPitch), cy=std::cos(gYaw), sy=std::sin(gYaw);
        Vec3 camPos{ target.x + gRadius*cp*sy, target.y + gRadius*sp, target.z + gRadius*cp*cy };

        Mat4 proj = matPerspective(45.0f, (float)fbw/(float)fbh, 0.1f, 200.0f);
        Mat4 view = matLookAt(camPos, target, Vec3{0,1,0});
        Mat4 viewProj = matMul(proj, view);

        // linhas
        glUseProgram(progLines);
        glUniformMatrix4fv(uVP_L,1,GL_FALSE,viewProj.m);
        glUniform3f(uColL,0.6f,0.6f,0.65f);
        glBindVertexArray(vaoLines);
        glDrawArrays(GL_LINES, 0, (GLsizei)(lineVerts.size()/3));
        glBindVertexArray(0);

        // esferas
        glUseProgram(progSphere);
        glUniformMatrix4fv(uVP_S,1,GL_FALSE,viewProj.m);
        glUniform3f(uLight, lightDir.x, lightDir.y, lightDir.z);
        glUniform3f(uCam, camPos.x, camPos.y, camPos.z);
        glBindVertexArray(vaoSphere);

        for(int s=0;s<NUM_STAGES;++s){
            for(size_t b=0;b<stages[s].size();++b){
                for(size_t n=0;n<stages[s][b].size();++n){
                    float rgb[3]; colorForCount(stages[s][b][n], rgb);
                    Mat4 model = matMul(matTranslate(pos[s][b][n]), matScale(radiusSphere));
                    glUniformMatrix4fv(uModelS,1,GL_FALSE,model.m);
                    glUniform3f(uBase, rgb[0], rgb[1], rgb[2]);
                    glDrawElements(GL_TRIANGLES, (GLsizei)sphereIdx.size(), GL_UNSIGNED_INT, 0);
                }
            }
        }
        glBindVertexArray(0);

        glfwSwapBuffers(win);
    }

    glDeleteVertexArrays(1,&vaoSphere);
    glDeleteBuffers(1,&vboSphere);
    glDeleteBuffers(1,&eboSphere);
    glDeleteVertexArrays(1,&vaoLines);
    glDeleteBuffers(1,&vboLines);
    glDeleteProgram(progLines);
    glDeleteProgram(progSphere);
    glfwDestroyWindow(win);
    glfwTerminate();
    return EXIT_SUCCESS;
}
