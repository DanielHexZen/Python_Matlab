import control as ct
import matplotlib.pyplot as plt
import numpy as np
import sys

# --- FUNKTION: Asymptotischer Bode-Plot (Logik aus Matlab portiert) ---
def plot_asymp(G):
    print(">> Berechne asymptotischen Verlauf...")
    
    # Zerlegung in Pole und Nullstellen
    p = G.poles()
    z = G.zeros()
    k = G.dcgain() # Vorsicht: Das ist inf bei Integratoren
    
    # Gain aus Transferfunktion extrahieren (high frequency gain factor)
    # control library speichert numerator/denominator als listen
    num = G.num[0][0]
    den = G.den[0][0]
    sys_k = num[0] / den[0] # Das 'k' aus zpk

    # -- Robuste K_Bode Berechnung --
    tol = 1e-10
    idx_p_0 = np.abs(p) < tol
    idx_z_0 = np.abs(z) < tol
    num_int = np.sum(idx_p_0)
    num_diff = np.sum(idx_z_0)
    
    p_rest = p[~idx_p_0]
    z_rest = z[~idx_z_0]
    
    # Produkt der Nullstellen/Pole (negativ)
    prod_z = np.prod(-z_rest) if len(z_rest) > 0 else 1.0
    prod_p = np.prod(-p_rest) if len(p_rest) > 0 else 1.0
    
    # Der Bode-Gain (Startwert für s->0 ohne Integratoren)
    K_bode = np.real(sys_k * prod_z / prod_p)
    
    # -- Frequenzbereich bestimmen --
    corners = np.concatenate([np.abs(z_rest), np.abs(p_rest)])
    corners = corners[corners > 0]
    
    if len(corners) == 0:
        w_min, w_max = -2, 2
    else:
        w_min = np.floor(np.log10(np.min(corners))) - 2
        w_max = np.ceil(np.log10(np.max(corners))) + 2
        
    w = np.logspace(w_min, w_max, 1000)
    
    # -- Amplitude --
    mag_db = 20 * np.log10(np.abs(K_bode)) * np.ones_like(w)
    
    # Integratoren/Differenzierer Einfluss auf Steigung
    net_slope = num_diff - num_int
    mag_db += 20 * net_slope * np.log10(w)
    
    # Pole
    for pole in p_rest:
        wc = np.abs(pole)
        mag_db -= 20 * np.log10(np.maximum(1, w/wc))
        
    # Nullstellen
    for zero in z_rest:
        wc = np.abs(zero)
        mag_db += 20 * np.log10(np.maximum(1, w/wc))
        
    # -- Phase --
    phase = np.zeros_like(w)
    
    if K_bode < 0:
        phase += 180
        
    phase += 90 * (num_diff - num_int)
    
    # Pole Phase
    for pole in p_rest:
        wc = np.abs(pole)
        # Stabil (Real < 0) -> -90, Instabil -> +90
        direction = -1 if np.real(pole) <= 0 else 1
        phase[w >= wc] += 90 * direction
        
    # Nullstellen Phase
    for zero in z_rest:
        wc = np.abs(zero)
        direction = 1 if np.real(zero) <= 0 else -1
        phase[w >= wc] += 90 * direction
        
    # -- PLOTTEN --
    plt.figure(figsize=(10, 8))
    
    # Magnitude
    plt.subplot(2, 1, 1)
    plt.semilogx(w, mag_db, 'b', linewidth=2, label='Asymptote')
    # Exakte Kurve zum Vergleich
    mag_real, phase_real, w_real = ct.bode(G, w, plot=False)
    plt.semilogx(w_real, 20*np.log10(mag_real), 'r--', alpha=0.6, label='Exakt')
    plt.grid(which='both', linestyle='-', alpha=0.5)
    plt.ylabel('Amplitude [dB]')
    plt.title(f'Bode-Diagramm (K_Bode = {K_bode:.3g})')
    plt.legend()
    
    # Phase
    plt.subplot(2, 1, 2)
    plt.semilogx(w, phase, 'b', linewidth=2, label='Asymptote')
    plt.semilogx(w_real, np.degrees(phase_real), 'r--', alpha=0.6, label='Exakt')
    plt.grid(which='both', linestyle='-', alpha=0.5)
    plt.ylabel('Phase [deg]')
    plt.xlabel('Frequenz [rad/s]')
    plt.yticks(np.arange(-270, 271, 45))
    
    print(">> Plot geöffnet. Schließe das Fenster, um fortzufahren.")
    plt.show()

# --- HELFER: Eingabe parsen ---
def parse_input(prompt_text):
    while True:
        try:
            user_in = input(prompt_text)
            # Entferne Klammern und Kommas für Flexibilität
            clean_in = user_in.replace('[', '').replace(']', '').replace(',', ' ')
            parts = clean_in.split()
            nums = [float(x) for x in parts]
            if len(nums) == 0: continue
            return nums
        except ValueError:
            print("❌ Fehler: Bitte Zahlen eingeben (z.B. 1 2 3)")

# --- MAIN LOOP ---
def main():
    print("=========================================")
    print("    Python Control Tool (Matlab-Style)   ")
    print("=========================================")
    
    current_sys = None
    
    while True:
        if current_sys is None:
            print("\n--- Neue Übertragungsfunktion definieren ---")
            print("Format: Koeffizienten wie in Matlab (z.B. [1 0] für 's')")
            num = parse_input("Zähler (Numerator): ")
            den = parse_input("Nenner (Denominator): ")
            
            # System erstellen
            current_sys = ct.TransferFunction(num, den)
            print("\n✅ System definiert:")
            print("-" * 30)
            print(current_sys)
            print("-" * 30)
        
        # Befehlsauswahl
        cmd = input("\nBefehl (bode / asymp / new / exit): ").strip().lower()
        
        if cmd == 'bode':
            print(">> Erstelle Standard Bode-Plot...")
            plt.figure(figsize=(10, 6))
            ct.bode_plot(current_sys, dB=True)
            plt.grid(which='both')
            plt.show()
            
        elif cmd == 'asymp':
            plot_asymp(current_sys)
            
        elif cmd == 'new':
            current_sys = None # Reset
            
        elif cmd == 'exit':
            print("Bye!")
            sys.exit()
            
        else:
            print("❌ Unbekannter Befehl.")

if __name__ == "__main__":
    main()