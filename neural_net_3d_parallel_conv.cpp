// neural_net_3d_parallel_conv.cpp (rev)
// g++ -std=c++17 -O2 -I include src\glad.c neural_net_3d_parallel_conv.cpp -L lib -lglfw3dll -lopengl32 -lgdi32 -luser32 -lkernel32 -o neural3d.exe

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

//================ Shaders ================
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
static const char* LIT_VS = R"(#version 330 core
layout (location=0) in vec3 aPos;
layout (location=1) in vec3 aNormal;
uniform mat4 uModel;
uniform mat4 uViewProj;
out vec3 vWorldPos; out vec3 vNormal;
void main(){
    vec4 wp = uModel * vec4(aPos,1.0);
    vWorldPos = wp.xyz;
    mat3 normalMat = transpose(inverse(mat3(uModel)));
    vNormal = normalize(normalMat * aNormal);
    gl_Position = uViewProj * wp;
}
)";
static const char* LIT_FS = R"(#version 330 core
in vec3 vWorldPos; in vec3 vNormal; out vec4 FragColor;
uniform vec3 uBaseColor, uLightDir, uCamPos;
void main(){
    vec3 n = normalize(vNormal);
    vec3 L = normalize(uLightDir);
    vec3 V = normalize(uCamPos - vWorldPos);
    float diff = max(dot(n,L),0.0);
    float rim  = pow(1.0 - max(dot(n,V),0.0), 2.0);
    vec3 col = mix(uBaseColor*0.55, uBaseColor*1.15, diff) + rim*0.25;
    FragColor = vec4(clamp(col,0.0,1.0),1.0);
}
)";

//================ GL helpers ================
static void glfwErrorCb(int c,const char* d){ std::fprintf(stderr,"GLFW %d: %s\n",c,d); }
static GLuint compile(GLenum t,const char* s){ GLuint sh=glCreateShader(t); glShaderSource(sh,1,&s,nullptr); glCompileShader(sh); GLint ok; glGetShaderiv(sh,GL_COMPILE_STATUS,&ok); if(!ok){ GLint n; glGetShaderiv(sh,GL_INFO_LOG_LENGTH,&n); std::string log(n,'\0'); glGetShaderInfoLog(sh,n,nullptr,log.data()); std::fprintf(stderr,"%s\n",log.c_str()); std::exit(1);} return sh; }
static GLuint linkProgram(const char* vs,const char* fs){ GLuint v=compile(GL_VERTEX_SHADER,vs), f=compile(GL_FRAGMENT_SHADER,fs); GLuint p=glCreateProgram(); glAttachShader(p,v); glAttachShader(p,f); glLinkProgram(p); glDeleteShader(v); glDeleteShader(f); GLint ok; glGetProgramiv(p,GL_LINK_STATUS,&ok); if(!ok){ GLint n; glGetProgramiv(p,GL_INFO_LOG_LENGTH,&n); std::string log(n,'\0'); glGetProgramInfoLog(p,n,nullptr,log.data()); std::fprintf(stderr,"%s\n",log.c_str()); std::exit(1);} return p; }

//================ Math ================
struct Mat4{ float m[16]; };
static Mat4 matIdentity(){ Mat4 r{}; r.m[0]=r.m[5]=r.m[10]=r.m[15]=1; return r; }
static Mat4 matMul(const Mat4& a,const Mat4& b){ Mat4 r{}; for(int c=0;c<4;++c) for(int rI=0;rI<4;++rI) r.m[c*4+rI]=a.m[0*4+rI]*b.m[c*4+0]+a.m[1*4+rI]*b.m[c*4+1]+a.m[2*4+rI]*b.m[c*4+2]+a.m[3*4+rI]*b.m[c*4+3]; return r; }
static Mat4 matTranslate(const Vec3& t){ Mat4 r=matIdentity(); r.m[12]=t.x; r.m[13]=t.y; r.m[14]=t.z; return r; }
static Mat4 matScale(float sx,float sy,float sz){ Mat4 r{}; r.m[0]=sx; r.m[5]=sy; r.m[10]=sz; r.m[15]=1; return r; }
static Mat4 matPerspective(float fovy,float aspect,float n,float f){ float k=1.0f/std::tan(fovy*PI/180.0f/2.0f); Mat4 r{}; r.m[0]=k/aspect; r.m[5]=k; r.m[10]=(f+n)/(n-f); r.m[11]=-1; r.m[14]=(2*f*n)/(n-f); return r; }
static Vec3 normalize(const Vec3& v){ float l=std::sqrt(v.x*v.x+v.y*v.y+v.z*v.z); return {v.x/l,v.y/l,v.z/l}; }
static Vec3 cross(const Vec3& a,const Vec3& b){ return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }
static float dotv(const Vec3& a,const Vec3& b){ return a.x*b.x+a.y*b.y+a.z*b.z; }
static Mat4 matLookAt(const Vec3& eye,const Vec3& c,const Vec3& up){ Vec3 f=normalize({c.x-eye.x,c.y-eye.y,c.z-eye.z}); Vec3 s=normalize(cross(f,up)); Vec3 u=cross(s,f); Mat4 r=matIdentity(); r.m[0]=s.x; r.m[4]=s.y; r.m[8]=s.z; r.m[1]=u.x; r.m[5]=u.y; r.m[9]=u.z; r.m[2]=-f.x; r.m[6]=-f.y; r.m[10]=-f.z; r.m[12]=-dotv(s,eye); r.m[13]=-dotv(u,eye); r.m[14]= dotv(f,eye); return r; }

//================ Entrada / parser ================
static std::string readAllStdin(){ std::ios::sync_with_stdio(false); std::cin.tie(nullptr); std::string s,l; while(std::getline(std::cin,l)){ s+=l; s+=' '; } return s; }
static std::string readAllFile(const char* p){ std::ifstream f(p,std::ios::binary); if(!f) return ""; return std::string((std::istreambuf_iterator<char>(f)),{}); }

struct Node{ enum{NUM,ARR,CONV} kind=NUM; long long val=0; std::vector<Node> arr; int A=0,B=0,C=0; };
static void skipWS(const std::string& s,size_t& i){ while(i<s.size() && (s[i]==' '||s[i]=='\n'||s[i]=='\t'||s[i]=='\r'||s[i]==',')) ++i; }
static bool parseNumberToken(const std::string& s,size_t& i,long long& out){ skipWS(s,i); bool neg=false; if(i<s.size()&&s[i]=='-'){neg=true;++i;} if(i>=s.size()||!(s[i]>='0'&&s[i]<='9')) return false; long long a=0; while(i<s.size()&&(s[i]>='0'&&s[i]<='9')){ a=a*10+(s[i]-'0'); ++i;} out=neg?-a:a; return true; }
static bool parseConv(const std::string& s,size_t& i,int& A,int& B,int& C){ skipWS(s,i); if(i>=s.size()||s[i]!='{') return false; ++i; long long a=0,b=0,c=0; if(!parseNumberToken(s,i,a)) return false; if(i>=s.size()||(s[i]!='x'&&s[i]!='X')) return false; ++i; if(!parseNumberToken(s,i,b)) return false; if(i>=s.size()||(s[i]!='x'&&s[i]!='X')) return false; ++i; if(!parseNumberToken(s,i,c)) return false; skipWS(s,i); if(i>=s.size()||s[i]!='}') return false; ++i; A=(int)std::max(0LL,a); B=(int)std::max(0LL,b); C=(int)std::max(0LL,c); return true; }
static Node parseNode(const std::string& s,size_t& i){
    skipWS(s,i);
    if(i>=s.size()) return {};
    if(s[i]=='{'){ int A,B,C; size_t j=i; if(!parseConv(s,j,A,B,C)) return {}; i=j; Node n; n.kind=Node::CONV; n.A=A; n.B=B; n.C=C; return n; }
    if(s[i]=='['){ ++i; Node n; n.kind=Node::ARR; while(true){ skipWS(s,i); if(i>=s.size()) break; if(s[i]==']'){ ++i; break; } n.arr.push_back(parseNode(s,i)); skipWS(s,i); if(i<s.size()&&s[i]==']'){ ++i; break; } if(i<s.size()&&s[i]==',') ++i; } return n; }
    long long v=0; if(!parseNumberToken(s,i,v)) return {}; Node n; n.kind=Node::NUM; n.val=v; return n;
}

struct Unit{ enum Type{Neuron,Conv} type=Neuron; int count=0; int A=0,B=0,C=0; };
using Branch = std::vector<Unit>;
using Stage  = std::vector<Branch>;

static bool nodeIsAllNums(const Node& n){ for(const auto& c:n.arr) if(c.kind!=Node::NUM) return false; return true; }
static bool nodeIsAllArraysOfNumsOrConvs(const Node& n){
    for(const auto& c:n.arr){ if(c.kind==Node::ARR){ if(!nodeIsAllNums(c)) return false; } else if(c.kind==Node::CONV){} else return false; }
    return true;
}
static std::vector<Stage> toStages(const Node& root){
    std::vector<Stage> S; if(root.kind!=Node::ARR) return S;
    for(const auto& s : root.arr){
        Stage stage;
        if(s.kind==Node::CONV){ Branch b; b.push_back(Unit{Unit::Conv,0,s.A,s.B,s.C}); stage.push_back(std::move(b)); }
        else if(s.kind==Node::ARR && nodeIsAllNums(s)){ Branch b; for(const auto& e:s.arr) b.push_back(Unit{Unit::Neuron,(int)std::max(0LL,e.val),0,0,0}); stage.push_back(std::move(b)); }
        else if(s.kind==Node::ARR && nodeIsAllArraysOfNumsOrConvs(s)){
            for(const auto& it:s.arr){ Branch b; if(it.kind==Node::CONV) b.push_back(Unit{Unit::Conv,0,it.A,it.B,it.C}); else for(const auto& e:it.arr) b.push_back(Unit{Unit::Neuron,(int)std::max(0LL,e.val),0,0,0}); stage.push_back(std::move(b)); }
        }else{ Branch b; if(s.kind==Node::ARR){ for(const auto& e:s.arr){ if(e.kind==Node::NUM) b.push_back(Unit{Unit::Neuron,(int)std::max(0LL,e.val),0,0,0}); if(e.kind==Node::CONV) b.push_back(Unit{Unit::Conv,0,e.A,e.B,e.C}); } } if(b.empty()) b.push_back(Unit{Unit::Neuron,0,0,0,0}); stage.push_back(std::move(b)); }
        S.push_back(std::move(stage));
    }
    if(S.empty()){
        S = { Stage{ Branch{ {Unit::Neuron,0},{Unit::Neuron,0},{Unit::Neuron,0},{Unit::Neuron,0},{Unit::Neuron,0} } },
              Stage{ Branch{ {Unit::Neuron,0},{Unit::Neuron,0},{Unit::Neuron,0},{Unit::Neuron,0} } },
              Stage{ Branch{ {Unit::Neuron,0},{Unit::Neuron,0},{Unit::Neuron,0} } },
              Stage{ Branch{ {Unit::Neuron,0} } } };
    }
    return S;
}

//================ Cores =================
static void colorForCount(int c,float o[3]){
    if(c<=0){ o[0]=0.85f;o[1]=0.85f;o[2]=0.85f; return; } // cinza
    if(c==1){ o[0]=0.20f;o[1]=0.40f;o[2]=1.00f; return; } // azul
    if(c==2){ o[0]=0.20f;o[1]=1.00f;o[2]=0.20f; return; } // verde
    if(c==3){ o[0]=1.00f;o[1]=0.95f;o[2]=0.10f; return; } // amarelo
    if(c==4){ o[0]=1.00f;o[1]=0.60f;o[2]=0.10f; return; } // laranja
    o[0]=1.00f;o[1]=0.25f;o[2]=0.25f;                         // vermelho
}

//================ Geometrias ================
static void buildSphereMesh(int stacks,int slices,std::vector<float>& v,std::vector<unsigned int>& idx){
    for(int i=0;i<=stacks;++i){ float vv=(float)i/stacks, phi=vv*PI; float y=std::cos(phi), r=std::sin(phi);
        for(int j=0;j<=slices;++j){ float uu=(float)j/slices, th=uu*2*PI; float x=r*std::cos(th), z=r*std::sin(th);
            v.insert(v.end(),{x,y,z,x,y,z}); } }
    int stride=slices+1; for(int i=0;i<stacks;++i) for(int j=0;j<slices;++j){ unsigned a=i*stride+j,b=(i+1)*stride+j,c=a+1,d=b+1; idx.insert(idx.end(),{a,b,c,b,d,c}); }
}
static void buildUnitCube(std::vector<float>& v,std::vector<unsigned int>& idx){
    struct V{float x,y,z,nx,ny,nz;};
    std::vector<V> vs={
        {-0.5f,-0.5f, 0.5f,0,0,1},{0.5f,-0.5f, 0.5f,0,0,1},{0.5f,0.5f,0.5f,0,0,1},{-0.5f,0.5f,0.5f,0,0,1},
        {-0.5f,-0.5f,-0.5f,0,0,-1},{-0.5f,0.5f,-0.5f,0,0,-1},{0.5f,0.5f,-0.5f,0,0,-1},{0.5f,-0.5f,-0.5f,0,0,-1},
        {-0.5f,-0.5f,-0.5f,-1,0,0},{-0.5f,-0.5f,0.5f,-1,0,0},{-0.5f,0.5f,0.5f,-1,0,0},{-0.5f,0.5f,-0.5f,-1,0,0},
        {0.5f,-0.5f,-0.5f,1,0,0},{0.5f,0.5f,-0.5f,1,0,0},{0.5f,0.5f,0.5f,1,0,0},{0.5f,-0.5f,0.5f,1,0,0},
        {-0.5f,-0.5f,-0.5f,0,-1,0},{0.5f,-0.5f,-0.5f,0,-1,0},{0.5f,-0.5f,0.5f,0,-1,0},{-0.5f,-0.5f,0.5f,0,-1,0},
        {-0.5f,0.5f,-0.5f,0,1,0},{-0.5f,0.5f,0.5f,0,1,0},{0.5f,0.5f,0.5f,0,1,0},{0.5f,0.5f,-0.5f,0,1,0}
    };
    unsigned int id[]={0,1,2,0,2,3, 4,5,6,4,6,7, 8,9,10,8,10,11, 12,13,14,12,14,15, 16,17,18,16,18,19, 20,21,22,20,22,23};
    for(auto& p:vs) v.insert(v.end(),{p.x,p.y,p.z,p.nx,p.ny,p.nz});
    idx.insert(idx.end(),std::begin(id),std::end(id));
}

//================ Layout / posições ================
struct UnitPos { Vec3 center; bool isConv; int A,B,C; float w,h,d; int count; };
using StageUnits = std::vector<std::vector<std::vector<UnitPos>>>;

static StageUnits computePositionsAndDims(
    const std::vector<Stage>& stages,
    float& spanX,float& spanY,float& spanZ, Vec3& centerOut)
{
    int maxA=1,maxB=1,maxC=1;
    for(const auto& st:stages) for(const auto& br:st) for(const auto& u:br)
        if(u.type==Unit::Conv){ maxA=std::max(maxA,u.A); maxB=std::max(maxB,u.B); maxC=std::max(maxC,u.C); }

    const float Wmax=3.4f,Hmax=3.4f,Dmax=1.8f;
    const float ySpacing=2.1f, zBranchSpacing=2.5f;
    const float x0=-6.5f,x1=6.5f;
    const int L=(int)stages.size(); const float dx=(L>1)?(x1-x0)/(L-1):0.0f;

    StageUnits out(L);
    float minX=1e9f,maxX=-1e9f,minY=1e9f,maxY=-1e9f,minZ=1e9f,maxZ=-1e9f;

    for(int s=0;s<L;++s){
        int B=(int)stages[s].size(); out[s].resize(B);
        float zStart=-0.5f*(B-1)*zBranchSpacing;
        for(int b=0;b<B;++b){
            int N=(int)stages[s][b].size(); out[s][b].resize(N);
            float yStart=-0.5f*(N-1)*ySpacing;
            for(int i=0;i<N;++i){
                const Unit& u=stages[s][b][i]; UnitPos up{};
                up.center={x0+s*dx, yStart+i*ySpacing, zStart+b*zBranchSpacing};
                if(u.type==Unit::Conv){
                    up.isConv=true;
                    up.w=std::max(0.6f, Wmax*(float)u.A/(float)maxA);
                    up.h=std::max(0.6f, Hmax*(float)u.B/(float)maxB);
                    up.d=std::max(0.35f, Dmax*(float)u.C/(float)maxC);
                    up.A=u.A; up.B=u.B; up.C=u.C; up.count=0;
                }else{
                    up.isConv=false; up.w=up.h=up.d=0; up.count=u.count;
                }
                out[s][b][i]=up;
                // meia-extensão por eixo: conv => X=d/2, Y=h/2, Z=w/2 ; neurônio => 0.7
                float hx = up.isConv ? (up.d * 0.5f) : 0.7f;
                float hy = up.isConv ? (up.h * 0.5f) : 0.7f;
                float hz = up.isConv ? (up.w * 0.5f) : 0.7f;


                minX = std::min(minX, up.center.x - hx);  maxX = std::max(maxX, up.center.x + hx);
                minY = std::min(minY, up.center.y - hy);  maxY = std::max(maxY, up.center.y + hy);
                minZ = std::min(minZ, up.center.z - hz);  maxZ = std::max(maxZ, up.center.z + hz);

            }
        }
    }
    spanX=maxX-minX; spanY=maxY-minY; spanZ=maxZ-minZ;
    centerOut = { (minX+maxX)*0.5f, (minY+maxY)*0.5f, (minZ+maxZ)*0.5f };
    return out;
}

// pontos de ancoragem (para linhas) — evitam “passar por dentro”
static Vec3 anchorTowardsNext(const UnitPos& u){ // sai no lado direito
    const float r=0.7f;
    if(u.isConv) return {u.center.x + u.d*0.5f + 0.02f, u.center.y, u.center.z};
    return {u.center.x + r, u.center.y, u.center.z};
}
static Vec3 anchorFromPrev(const UnitPos& u){    // entra no lado esquerdo
    const float r=0.7f;
    if(u.isConv) return {u.center.x - u.d*0.5f - 0.02f, u.center.y, u.center.z};
    return {u.center.x - r, u.center.y, u.center.z};
}


//================ Câmera orbit ================
static float gYaw=0.0f, gPitch=0.0f, gRadius=20.0f;
static bool  gRot=false; static double gLastX=0, gLastY=0;
static Vec3  gTarget{0,0,0};

static void mouseButtonCallback(GLFWwindow* w,int button,int action,int){
    if(button==GLFW_MOUSE_BUTTON_LEFT){
        if(action==GLFW_PRESS){ gRot=true; glfwGetCursorPos(w,&gLastX,&gLastY); }
        else if(action==GLFW_RELEASE){ gRot=false; }
    }
}
static void cursorPosCallback(GLFWwindow*,double x,double y){
    if(!gRot) return;
    double dx=x-gLastX, dy=y-gLastY; gLastX=x; gLastY=y;
    const float sens=0.005f;
    gYaw   += (float)dx*sens;
    gPitch += (float)dy*sens;
    float lim=PI/2.0f-0.05f;
    if(gPitch> lim) gPitch= lim;
    if(gPitch<-lim) gPitch=-lim;
}
static void scrollCallback(GLFWwindow*, double, double yoff){
    if(yoff>0) gRadius*=0.9f; else gRadius*=1.1f;
    if(gRadius<5.0f) gRadius=5.0f;
    if(gRadius>300.0f) gRadius=300.0f;
}

//================ Main ================
int main(int argc,char** argv){
    std::string raw = (argc>1)? readAllFile(argv[1]) : readAllStdin();
    if(raw.empty()) raw = "[{224x224x64}, [3,2,1], [1]]"; // conv primeiro, por padrão

    size_t i=0; Node root=parseNode(raw,i);
    auto stages = toStages(root);

    glfwSetErrorCallback(glfwErrorCb);
    if(!glfwInit()){ std::fprintf(stderr,"Falha GLFW\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* win = glfwCreateWindow(1000,700,"Neural Net 3D - Conv primeiro + Orbit",nullptr,nullptr);
    if(!win){ glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win); glfwSwapInterval(1);
    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){ std::fprintf(stderr,"Falha GLAD\n"); return 1; }

    glfwSetMouseButtonCallback(win, mouseButtonCallback);
    glfwSetCursorPosCallback(win, cursorPosCallback);
    glfwSetScrollCallback(win, scrollCallback);

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.07f,0.08f,0.11f,1.0f);

    GLuint progLines = linkProgram(LINE_VS, LINE_FS);
    GLuint progLit   = linkProgram(LIT_VS,  LIT_FS);

    // esfera
    std::vector<float> sV; std::vector<unsigned int> sI;
    buildSphereMesh(24,32,sV,sI);
    GLuint vaoS=0,vboS=0,eboS=0;
    glGenVertexArrays(1,&vaoS); glGenBuffers(1,&vboS); glGenBuffers(1,&eboS);
    glBindVertexArray(vaoS);
    glBindBuffer(GL_ARRAY_BUFFER,vboS); glBufferData(GL_ARRAY_BUFFER,sV.size()*sizeof(float),sV.data(),GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,eboS); glBufferData(GL_ELEMENT_ARRAY_BUFFER,sI.size()*sizeof(unsigned int),sI.data(),GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,6*sizeof(float),(void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,6*sizeof(float),(void*)(3*sizeof(float)));
    glBindVertexArray(0);

    // cubo
    std::vector<float> cV; std::vector<unsigned int> cI;
    buildUnitCube(cV,cI);
    GLuint vaoC=0,vboC=0,eboC=0;
    glGenVertexArrays(1,&vaoC); glGenBuffers(1,&vboC); glGenBuffers(1,&eboC);
    glBindVertexArray(vaoC);
    glBindBuffer(GL_ARRAY_BUFFER,vboC); glBufferData(GL_ARRAY_BUFFER,cV.size()*sizeof(float),cV.data(),GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,eboC); glBufferData(GL_ELEMENT_ARRAY_BUFFER,cI.size()*sizeof(unsigned int),cI.data(),GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,6*sizeof(float),(void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,6*sizeof(float),(void*)(3*sizeof(float)));
    glBindVertexArray(0);

    // posições + centro do modelo
    float spanX,spanY,spanZ; Vec3 center;
    StageUnits pos = computePositionsAndDims(stages, spanX,spanY,spanZ, center);
    gTarget = center;                       // pivô no centro do modelo
    gRadius = std::max(12.0f, std::max({spanX,spanY,spanZ})*1.6f);

    // linhas entre estágios (usando âncoras)
    std::vector<float> lineVerts;
    for(size_t s=0;s+1<stages.size();++s)
        for(size_t b0=0;b0<stages[s].size();++b0)
            for(size_t i0=0;i0<stages[s][b0].size();++i0){
                Vec3 a = anchorTowardsNext(pos[s][b0][i0]);
                for(size_t b1=0;b1<stages[s+1].size();++b1)
                    for(size_t i1=0;i1<stages[s+1][b1].size();++i1){
                        Vec3 b = anchorFromPrev(pos[s+1][b1][i1]);
                        lineVerts.insert(lineVerts.end(),{a.x,a.y,a.z, b.x,b.y,b.z});
                    }
            }
    GLuint vaoL=0,vboL=0;
    glGenVertexArrays(1,&vaoL); glGenBuffers(1,&vboL);
    glBindVertexArray(vaoL);
    glBindBuffer(GL_ARRAY_BUFFER,vboL); glBufferData(GL_ARRAY_BUFFER,lineVerts.size()*sizeof(float),lineVerts.data(),GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,3*sizeof(float),(void*)0);
    glBindVertexArray(0);

    // uniforms
    GLint uVP_L = glGetUniformLocation(progLines,"uViewProj");
    GLint uColL = glGetUniformLocation(progLines,"uColor");
    GLint uVP   = glGetUniformLocation(progLit,"uViewProj");
    GLint uModel= glGetUniformLocation(progLit,"uModel");
    GLint uBase = glGetUniformLocation(progLit,"uBaseColor");
    GLint uLight= glGetUniformLocation(progLit,"uLightDir");
    GLint uCam  = glGetUniformLocation(progLit,"uCamPos");

    Vec3 lightDir = normalize(Vec3{0.6f,0.7f,0.5f});

    while(!glfwWindowShouldClose(win)){
        glfwPollEvents();
        // atalhos de view: 1 frontal, 2 lado dir, 3 lado esq, 0 reset
        if(glfwGetKey(win,GLFW_KEY_1)==GLFW_PRESS){ gYaw=0; gPitch=0; }
        if(glfwGetKey(win,GLFW_KEY_2)==GLFW_PRESS){ gYaw= PI/2; gPitch=0; }
        if(glfwGetKey(win,GLFW_KEY_3)==GLFW_PRESS){ gYaw=-PI/2; gPitch=0; }
        if(glfwGetKey(win,GLFW_KEY_0)==GLFW_PRESS){ gYaw=0; gPitch=0; gRadius=std::max(12.0f,std::max({spanX,spanY,spanZ})*1.6f); }
        if(glfwGetKey(win,GLFW_KEY_ESCAPE)==GLFW_PRESS) glfwSetWindowShouldClose(win,GL_TRUE);

        int fbw,fbh; glfwGetFramebufferSize(win,&fbw,&fbh);
        glViewport(0,0,fbw,fbh);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

        float cp=std::cos(gPitch), sp=std::sin(gPitch), cy=std::cos(gYaw), sy=std::sin(gYaw);
        Vec3 cam{ gTarget.x + gRadius*cp*sy, gTarget.y + gRadius*sp, gTarget.z + gRadius*cp*cy };

        Mat4 proj=matPerspective(45.0f,(float)fbw/(float)fbh,0.1f,300.0f);
        Mat4 view=matLookAt(cam,gTarget,{0,1,0});
        Mat4 vp  =matMul(proj,view);

        // linhas (com depth test para oclusão correta, mas agora saem das faces)
        glUseProgram(progLines);
        glUniformMatrix4fv(uVP_L,1,GL_FALSE,vp.m);
        glUniform3f(uColL,0.6f,0.6f,0.65f);
        glBindVertexArray(vaoL);
        glLineWidth(1.2f);
        glDrawArrays(GL_LINES,0,(GLsizei)(lineVerts.size()/3));
        glBindVertexArray(0);

        // objetos
        glUseProgram(progLit);
        glUniformMatrix4fv(uVP,1,GL_FALSE,vp.m);
        glUniform3f(uLight,lightDir.x,lightDir.y,lightDir.z);
        glUniform3f(uCam,cam.x,cam.y,cam.z);

        // neurônios
        glBindVertexArray(vaoS);
        for(size_t s=0;s<stages.size();++s) for(size_t b=0;b<stages[s].size();++b) for(size_t n=0;n<stages[s][b].size();++n){
            const UnitPos& u=pos[s][b][n]; if(u.isConv) continue;
            float rgb[3]; colorForCount(u.count,rgb);
            Mat4 M=matMul(matTranslate(u.center),matScale(0.7f,0.7f,0.7f));
            glUniformMatrix4fv(uModel,1,GL_FALSE,M.m); glUniform3f(uBase,rgb[0],rgb[1],rgb[2]);
            glDrawElements(GL_TRIANGLES,(GLsizei)sI.size(),GL_UNSIGNED_INT,0);
        }
        glBindVertexArray(0);

        // convs (fatias)
// convs (blocos de fatias no eixo X: profundidade = d = C)
        glBindVertexArray(vaoC);
        for(size_t s=0;s<stages.size();++s){
            for(size_t b=0;b<stages[s].size();++b){
                for(size_t i=0;i<stages[s][b].size();++i){
                    const UnitPos& up = pos[s][b][i];
                    if(!up.isConv) continue;

                    // nº de “fatias” visuais conforme C
                    int panels = std::max(3, std::min(7, (up.C<=0?3: (3 + (up.C%5)))));
                    float totalX = up.d;                   // profundidade agora é no X
                    float t = totalX / (float)panels;      // espessura de cada fatia (X)
                    float x0 = -0.5f*totalX + 0.5f*t;      // centro da 1ª fatia

                    // A×B (up.w × up.h) é a face frontal (plano YZ)
                    float base[3] = {0.30f, 0.70f, 1.00f};
                    for(int k=0;k<panels;++k){
                        float xOff = x0 + k*t;
                        Mat4 M = matMul(
                                    matTranslate({up.center.x + xOff, up.center.y, up.center.z}),
                                    // escala: X=t (profundidade), Y=h (B), Z=w (A)
                                    matScale(t*0.95f, up.h, up.w));
                        glUniformMatrix4fv(uModel,1,GL_FALSE,M.m);
                        glUniform3f(uBase, base[0], base[1], base[2]);
                        glDrawElements(GL_TRIANGLES, (GLsizei)cI.size(), GL_UNSIGNED_INT, 0);
                            }
                        }
                    }
                }
        glBindVertexArray(0);


        glfwSwapBuffers(win);
    }

    glDeleteVertexArrays(1,&vaoS); glDeleteBuffers(1,&vboS); glDeleteBuffers(1,&eboS);
    glDeleteVertexArrays(1,&vaoC); glDeleteBuffers(1,&vboC); glDeleteBuffers(1,&eboC);
    glDeleteVertexArrays(1,&vaoL); glDeleteBuffers(1,&vboL);
    glDeleteProgram(progLines); glDeleteProgram(progLit);
    glfwDestroyWindow(win); glfwTerminate();
    return 0;
}
