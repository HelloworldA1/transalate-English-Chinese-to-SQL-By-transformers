zidian_x = '<SOS>,<EOS>,<PAD>,0,1,2,3,4,5,6,7,8,9,q,w,e,r,t,y,u,i,o,p,a,s,d,f,g,h,j,k,l,z,x,c,v,b,n,m'
zidian_x = {word: i for i, word in enumerate(zidian_x.split(','))}

zidian_xr = [k for k, v in zidian_x.items()]
print(zidian_xr)

zidian_y = {k.upper(): v for k, v in zidian_x.items()}

zidian_yr = [k for k, v in zidian_y.items()]