#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W 64
#define H 64
#define N (W*H)
#define T 256
#define O 32

static uint8_t a[N],b[N],m[N],q[O],r[O],u[O],v[O];
static uint32_t z=1;

static uint32_t g(){z^=z<<13;z^=z>>17;z^=z<<5;return z;}
static int p(int x,int y){x=(x+W)%W;y=(y+H)%H;return y*W+x;}
static uint8_t h(int x,int y,uint8_t e){
    int i=p(x,y),n=p(x,y-1),s=p(x,y+1),w=p(x-1,y),k=p(x+1,y);
    uint8_t c=a[i],d=((a[n]<<1)|(a[s]<<3)|(a[w]<<5)|(a[k]<<7))^m[i]^e;
    uint8_t t=(uint8_t)((c+d+(c>>1)+(d<<1))^(c*(d|1))^(d*(c|1)));
    return (uint8_t)((t<<1)|(t>>7));
}
static void j(uint8_t e){
    for(int y=0;y<H;y++)for(int x=0;x<W;x++)b[p(x,y)]=h(x,y,(uint8_t)(e+(x==0||y==0||x==W-1||y==H-1?e:0)));
    for(int i=0;i<N;i++){
        uint8_t x=a[i],y=b[i],d=x^y;
        m[i]=(uint8_t)((m[i]+((d&1)?y:x)+(m[i]>>1))^(d*29u));
        a[i]=y;
    }
}
static uint8_t s(){
    uint8_t x=0;
    for(int i=0;i<O;i++)x^=(uint8_t)(a[p(W-1,i*2)] + (a[p(W-2,i*2+1)]<<1) + m[p(W-1-i,H-1)]);
    return x;
}
static void l(uint8_t x){memmove(q,q+1,O-1);q[O-1]=x;memmove(r,r+1,O-1);r[O-1]=s();}
static uint8_t f(){
    uint8_t x=0;
    for(int i=1;i<O;i++)x^=(uint8_t)((q[i]-q[i-1])^(r[i]-r[i-1])^(u[i]^v[i-1]));
    memmove(u,u+1,O-1);u[O-1]=q[O-1];memmove(v,v+1,O-1);v[O-1]=r[O-1];
    return x;
}

int main(int c,char**v0){
    uint32_t n=c>1?(uint32_t)strtoul(v0[1],0,10):0;
    if(!n)n=1u<<20;
    for(int i=0;i<N;i++){a[i]=(uint8_t)g();m[i]=(uint8_t)g();}
    uint8_t e=0;
    for(uint32_t t=0;t<n;t++){
        int k=getchar();
        if(k!=EOF)e=(uint8_t)k;
        j(e);
        l(e);
        if((t&T-1u)==T-1u){uint8_t x=f();putchar(x);fflush(stdout);}
    }
    return 0;
}
