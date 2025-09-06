import numpy as np
from scipy.io.wavfile import write

def fm_synth(f_port: float, f_mod: float=3, env_port: float=0.5, env_mod: float=1.0, duracao: float=5.0, f_amostragem: int=44100) -> np.ndarray:
    t = np.linspace(0, duracao, int(f_amostragem*duracao), endpoint=False)
    mod = env_mod * np.sin(2*np.pi*f_mod*t)
    y = env_port * np.sin(2*np.pi*f_port*t + mod)
    return y

# --- Envelopes básicos ---
def env_exp(duracao, f_amostragem, tau=3):
    t = np.linspace(0, duracao, int(f_amostragem*duracao), endpoint=False)
    return np.exp(-t*tau/duracao)

def env_attack_decay(duracao, f_amostragem, atk=0.1, dec=0.2):
    N = int(f_amostragem*duracao)
    atk_len = int(atk*N)
    dec_len = int(dec*N)
    sus_len = N - atk_len - dec_len
    env = np.concatenate([
        np.linspace(0,1,atk_len,endpoint=False),
        np.ones(sus_len),
        np.linspace(1,0,dec_len,endpoint=False)
    ])
    return env[:N]

# --- Instrumentos ---
def fm_sopro(f, duracao, f_amostragem):
    env_p = env_attack_decay(duracao, f_amostragem, atk=0.2, dec=0.2)
    env_m = 2 * env_attack_decay(duracao, f_amostragem, atk=0.2, dec=0.2)
    return fm_synth(f, 2*f, env_p, env_m, duracao, f_amostragem)

def fm_corda(f, duracao, f_amostragem):
    env_p = env_exp(duracao, f_amostragem, tau=5)
    env_m = 3 * env_exp(duracao, f_amostragem, tau=5)
    return fm_synth(f, 3*f, env_p, env_m, duracao, f_amostragem)

def fm_bate(f, dur, f_amostragem):
    dur = 0.5
    env_p = env_exp(dur, f_amostragem, tau=10)
    env_m = 5 * env_exp(dur, f_amostragem, tau=10)
    return fm_synth(f, 7*f, env_p, env_m, dur, f_amostragem)


map_notas = {
    'A': 0,  
    'B': 2,  
    'C': 3,  
    'D': 5,  
    'E': 7,  
    'F': 8,  
    'G': 10
}

def nota_para_freq(nome, acidente, oitava):
    base_freq = 55
    semitons = map_notas[nome]

    if acidente == 's':
        semitons += 1
    elif acidente == 'b':
        semitons -= 1

    semitons += 12 * (oitava - 1)

    return base_freq * (2 ** (semitons / 12))

def interpretar(seq, f_amostragem=44100):
    partes = seq.split(",")
    duracao = 1.0  # duração padrão
    instrumento = fm_sopro
    saida = np.array([], dtype=float)

    for parte in partes:
        if parte.startswith("T"):  
            duracao = float(parte[1:])  # duração padrão
        elif parte.startswith("I"):
            if parte == "Is": instrumento = fm_sopro
            elif parte == "Ic": instrumento = fm_corda
            elif parte == "Ip": instrumento = fm_bate
        else:
            # Nota no formato: Nome + acidente + oitava + duração + intensidade
            nome = parte[0]            # A, B, C, D, E, F, G
            acidente = parte[1]        # n, s, b
            oitava = int(parte[2])     # 1 a 8
            dur_den = int(parte[3])    # denominador da duração
            intensidade = parte[4]     # F ou f

            freq = nota_para_freq(nome, acidente, oitava)
            dur = duracao/dur_den
            amp = 1.0 if intensidade == "F" else 0.5

            y = instrumento(freq, dur, f_amostragem) * amp
            saida = np.concatenate([saida, y])
    
    return saida

s_corda = 'T1,Ic,Cn34f,Dn34f,En32f,Gn32F,Gn32f,En32f,Fn32f,Fn31F'
s_sopro =  'T1,Is,Cn34f,Dn34f,En32f,Gn32F,Gn32f,En32f,Fn32f,Fn31F'
s_bate  = 'T1,Ip,Cn22F,Cn24f,Cn84F,Cn52f,Cn22f,Cn24F,Cn54F,Cn82F'

def main():
    f_amostragem = 44100
    
    seq = input("Digite a sequência musical:\n> ")
    print(seq)

    y = interpretar(seq, f_amostragem)

    salvar = input("Deseja salvar em WAV? (s/n): ").lower()
    if salvar == "s":
        nome_arquivo = input("Nome do arquivo (sem .wav): ")
        write(f"{nome_arquivo}.wav", f_amostragem, (y*32767).astype(np.int16))
        print(f"Arquivo salvo como {nome_arquivo}.wav")

    tocar = input("Deseja tocar o som agora? (s/n): ").lower()
    if tocar == "s":
        import sounddevice as sd
        sd.play(y, f_amostragem)
        sd.wait()

if __name__ == "__main__":
    main()
