import tkinter as tk
from tkinter import filedialog, messagebox, ttk, colorchooser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.signal import savgol_filter
import os
from datetime import datetime
import re # Para extrair o delimitador

class LpgFilterApp:
    def __init__(self, master):
        """
        Configura a interface gráfica principal (GUI) do aplicativo.
        v11.0: Layout otimizado com controles à esquerda e gráfico expandido à direita.
        """
        self.master = master
        master.title("Filtro Savitzky-Golay (v11.0 - Layout Otimizado)")
        master.geometry("1200x800") # Janela inicial maior

        # --- Variáveis de Estado ---
        self.active_filename = None
        self.active_wavelength = None
        self.active_intensity = None
        self.active_filtered_intensity = None
        self.loaded_data = {}
        
        self.active_valley_wl = None
        self.active_valley_intensity = None
        self.log_filepath = None
        
        self.color_original = 'black'
        self.color_filtrado = 'red'
        
        self.include_annotation_var = tk.BooleanVar(value=True)
        self.include_range_var = tk.BooleanVar(value=True)

        # --- Estrutura Principal (Grid) ---
        # 1. Frame principal que segura as duas colunas
        main_frame = tk.Frame(master)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=0) # Coluna de controles (peso 0)
        main_frame.grid_columnconfigure(1, weight=1) # Coluna do gráfico (peso 1)

        # --- Coluna da Esquerda: Controles (com Scrollbar v12.0) ---
        
        # 1. Criar um Canvas e uma Scrollbar
        control_canvas_frame = tk.Frame(main_frame)
        control_canvas_frame.grid(row=0, column=0, sticky="ns", padx=(0, 5))
        control_canvas_frame.grid_rowconfigure(0, weight=1)
        
        control_canvas = tk.Canvas(control_canvas_frame, width=350, borderwidth=0, highlightthickness=0)
        control_scrollbar = ttk.Scrollbar(control_canvas_frame, orient="vertical", command=control_canvas.yview)
        control_canvas.configure(yscrollcommand=control_scrollbar.set)
        
        control_scrollbar.grid(row=0, column=1, sticky="ns")
        control_canvas.grid(row=0, column=0, sticky="ns")
        
        # 2. Este é o Frame que vai conter todos os widgets
        # Ele é colocado *dentro* do canvas
        self.control_frame = tk.Frame(control_canvas)
        
        # 3. Adiciona o frame ao canvas
        control_canvas.create_window((0, 0), window=self.control_frame, anchor="nw")

        # 4. Função para atualizar a scrollregion
        def update_scrollregion(event):
            control_canvas.configure(scrollregion=control_canvas.bbox("all"))
        
        self.control_frame.bind("<Configure>", update_scrollregion)
        
        # Adiciona o bind para scroll do mouse (Windows/Mac/Linux)
        def on_mousewheel(event):
            # No Windows, event.delta é +-120. No Linux, é +-4 ou 5 (button 4/5).
            if event.num == 4: # Linux scroll up
                control_canvas.yview_scroll(-1, "units")
            elif event.num == 5: # Linux scroll down
                control_canvas.yview_scroll(1, "units")
            else: # Windows/Mac
                # Garante que event.delta exista antes de dividir
                if hasattr(event, 'delta'):
                    control_canvas.yview_scroll(-1 * (event.delta // 120), "units")
                else:
                    # Fallback para outros sistemas se delta não estiver disponível
                    if event.delta > 0:
                        control_canvas.yview_scroll(-1, "units")
                    else:
                        control_canvas.yview_scroll(1, "units")


        # Bind em 'control_canvas' e 'self.control_frame' pode ser mais robusto
        for widget in (control_canvas, self.control_frame, root):
            widget.bind_all("<MouseWheel>", on_mousewheel, add='+') # Windows/Mac
            widget.bind_all("<Button-4>", on_mousewheel, add='+')   # Linux
            widget.bind_all("<Button-5>", on_mousewheel, add='+')   # Linux


        # --- Botão Carregar (agora dentro de self.control_frame) ---
        self.load_button = tk.Button(self.control_frame, text="Carregar Arquivo(s) (.txt)", command=self.load_files)
        self.load_button.pack(fill='x', pady=(0, 10))

        # --- Frame: Customização do Gráfico (v10) ---
        color_frame = tk.LabelFrame(self.control_frame, text="Customização do Gráfico")
        color_frame.pack(fill='x', pady=5)
        
        self.color_orig_button = tk.Button(color_frame, text="Cor Original", command=lambda: self.pick_color('original'))
        self.color_orig_button.pack(side='left', padx=5, expand=True)
        self.color_preview_orig = tk.Label(color_frame, text="", bg=self.color_original, relief='sunken', width=4)
        self.color_preview_orig.pack(side='left', padx=(0, 10))

        self.color_filt_button = tk.Button(color_frame, text="Cor Filtrada", command=lambda: self.pick_color('filtrado'))
        self.color_filt_button.pack(side='left', padx=5, expand=True)
        self.color_preview_filt = tk.Label(color_frame, text="", bg=self.color_filtrado, relief='sunken', width=4)
        self.color_preview_filt.pack(side='left', padx=(0, 5))

        # --- Frame: Controles de Filtro e Busca (v7) ---
        filter_frame = tk.LabelFrame(self.control_frame, text="Controles de Filtro e Busca")
        filter_frame.pack(fill='x', pady=5)

        filter_grid = tk.Frame(filter_frame)
        filter_grid.pack(fill='x', padx=5, pady=5)
        
        tk.Label(filter_grid, text="Janela (ímpar):").grid(row=0, column=0, sticky='w', pady=2)
        self.window_entry = tk.Entry(filter_grid, width=7)
        self.window_entry.insert(0, "21")
        self.window_entry.grid(row=0, column=1, sticky='w', padx=5)

        tk.Label(filter_grid, text="Ordem Polinômio:").grid(row=1, column=0, sticky='w', pady=2)
        self.order_entry = tk.Entry(filter_grid, width=7)
        self.order_entry.insert(0, "3")
        self.order_entry.grid(row=1, column=1, sticky='w', padx=5)
        
        self.normalize_var = tk.BooleanVar(value=False)
        self.normalize_check = tk.Checkbutton(filter_grid, text="Normalizar", variable=self.normalize_var)
        self.normalize_check.grid(row=0, column=2, sticky='w', padx=5)

        tk.Label(filter_grid, text="Buscar Vale de (nm):").grid(row=2, column=0, sticky='w', pady=(8,2))
        self.range_start_entry = tk.Entry(filter_grid, width=7)
        self.range_start_entry.grid(row=2, column=1, sticky='w', padx=5)

        tk.Label(filter_grid, text="Até (nm):").grid(row=3, column=0, sticky='w', pady=2)
        self.range_end_entry = tk.Entry(filter_grid, width=7)
        self.range_end_entry.grid(row=3, column=1, sticky='w', padx=5)

        self.process_button = tk.Button(filter_frame, text="Aplicar Filtro e Buscar Vale", command=self.process_and_plot, state='disabled')
        self.process_button.pack(fill='x', padx=5, pady=(5, 10))

        # --- Frame: Log de Vales (Resultados) (v6) ---
        log_frame = tk.LabelFrame(self.control_frame, text="Log de Vales (Resultados)")
        log_frame.pack(fill='x', pady=5)

        tk.Label(log_frame, text="Nome da Amostra:").pack(anchor='w', padx=5)
        self.sample_name_entry = tk.Entry(log_frame)
        self.sample_name_entry.pack(fill='x', padx=5, pady=(0, 5))

        self.set_log_button = tk.Button(log_frame, text="Definir Arquivo de Log (.xlsx/.csv)", command=self.set_log_file)
        self.set_log_button.pack(fill='x', padx=5, pady=5)

        self.log_valley_button = tk.Button(log_frame, text="Registrar Vale no Log", command=self.log_valley_data, state='disabled')
        self.log_valley_button.pack(fill='x', padx=5, pady=5)

        tk.Label(log_frame, text="Arquivo de Log:", anchor='w').pack(fill='x', padx=5, pady=(5,0))
        self.log_file_label = tk.Label(log_frame, text="Nenhum definido", fg="gray", anchor='w', justify='left', relief='sunken', borderwidth=1)
        self.log_file_label.pack(fill='x', padx=5, pady=(0, 5))

        # --- Frame: Arquivos de Espectro (v3) ---
        list_frame = tk.LabelFrame(self.control_frame, text="Arquivos de Espectro")
        list_frame.pack(fill='x', pady=5, expand=False) # Não expande verticalmente
        
        self.valley_info_label = tk.Label(list_frame, text="Vale do Espectro: N/A", fg="blue", font=("Helvetica", 10, "bold"))
        self.valley_info_label.pack(anchor='w', padx=5, pady=2)
        
        list_subframe = tk.Frame(list_frame, height=150) # Altura fixa para a lista
        list_subframe.pack(fill='x', expand=False)
        list_subframe.pack_propagate(False) # Impede que a lista controle o tamanho

        self.scrollbar = tk.Scrollbar(list_subframe, orient='vertical')
        self.file_listbox = tk.Listbox(list_subframe, yscrollcommand=self.scrollbar.set, exportselection=False)
        self.scrollbar.config(command=self.file_listbox.yview)
        self.scrollbar.pack(side='right', fill='y')
        self.file_listbox.pack(side='left', fill='both', expand=True)
        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_select)
        
        # --- Frame: Salvamento ---
        save_main_frame = tk.LabelFrame(self.control_frame, text="Opções de Salvamento")
        save_main_frame.pack(fill='x', pady=(10, 0))

        self.save_full_spectrum_button = tk.Button(save_main_frame, text="Salvar Espectro Completo (.xlsx/.csv)", command=self.save_full_spectrum, state='disabled')
        self.save_full_spectrum_button.pack(fill='x', padx=5, pady=5)

        self.save_full_image_button = tk.Button(save_main_frame, text="Salvar Imagem (Completa)", command=self.save_plot_image, state='disabled')
        self.save_full_image_button.pack(fill='x', padx=5, pady=5)

        # Sub-frame: Imagem Apenas do Filtro (v9)
        save_filtered_img_frame = tk.LabelFrame(save_main_frame, text="Imagem Apenas do Filtro")
        save_filtered_img_frame.pack(fill='x', padx=5, pady=5)

        self.save_filtered_image_button = tk.Button(save_filtered_img_frame, text="Salvar Imagem (Só Filtro)", command=self.save_filtered_plot_only, state='disabled')
        self.save_filtered_image_button.pack(fill='x', padx=5, pady=5)

        check_frame = tk.Frame(save_filtered_img_frame)
        check_frame.pack(fill='x')
        self.include_annotation_check = tk.Checkbutton(check_frame, text="Incluir Anotação (Seta)", variable=self.include_annotation_var)
        self.include_annotation_check.pack(side='left', padx=5)
        self.include_range_check = tk.Checkbutton(check_frame, text="Incluir Faixa (Linhas)", variable=self.include_range_var)
        self.include_range_check.pack(side='left', padx=5)


        # --- Coluna da Direita: Gráfico ---
        self.graph_frame = tk.Frame(main_frame, bg='white')
        self.graph_frame.grid(row=0, column=1, sticky="nsew")
        self.graph_frame.grid_rowconfigure(0, weight=1)
        self.graph_frame.grid_columnconfigure(0, weight=1)

        # Configuração do Matplotlib
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.graph_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.grid(row=1, column=0, sticky="ew")
        
        self.ax.set_title("Carregue um ou mais arquivos para começar")
        self.ax.set_xlabel("Comprimento de Onda (nm)")
        self.ax.set_ylabel("Potência (dB)")
        self.ax.grid(True, linestyle=':', alpha=0.7)
        self.fig.tight_layout() # Ajusta o layout para evitar sobreposição

    # ===================================================================
    # FUNÇÕES DE LÓGICA
    # ===================================================================

    def pick_color(self, target):
        """Abre o seletor de cores e atualiza o gráfico."""
        color_code = colorchooser.askcolor(title=f"Escolher cor para {target}")
        if color_code[1]: # Se o usuário não cancelou (color_code[1] é o hex)
            if target == 'original':
                self.color_original = color_code[1]
                self.color_preview_orig.config(bg=self.color_original)
            elif target == 'filtrado':
                self.color_filtrado = color_code[1]
                self.color_preview_filt.config(bg=self.color_filtrado)
            
            # Se já houver dados, replota com as novas cores
            if self.active_wavelength is not None:
                self.process_and_plot(re_plot_only=True)


    def detect_delimiter(self, filepath):
        """
        Lê a primeira linha de um arquivo para adivinhar o delimitador.
        Prefere ; sobre , sobre espaço.
        """
        try:
            with open(filepath, 'r') as f:
                first_line = f.readline()
            
            if ';' in first_line:
                return ';'
            
            # Regex para encontrar vírgula usada como decimal (ex: 0,001)
            # Se encontrar, provavelmente o separador não é vírgula
            if re.search(r'\d,\d', first_line):
                # Se tem vírgula decimal, o separador é provavelmente tab ou espaço
                return None # Deixa o numpy decidir (padrão)
                
            if ',' in first_line:
                return ','
                
            return None # Deixa o numpy decidir (padrão: espaço/tab)
        except Exception:
            return None # Fallback

    def load_files(self):
        """
        Abre uma caixa de diálogo para selecionar um ou mais arquivos .txt.
        Carrega todos na lista e no dicionário de dados.
        """
        filepaths = filedialog.askopenfilenames(
            title="Selecione o(s) arquivo(s) de espectro",
            filetypes=(("Arquivos de Texto", "*.txt"), ("Todos os arquivos", "*.*"))
        )
        if not filepaths:
            return

        self.reset_data(clear_plot=False)
        
        try:
            for filepath in filepaths:
                delimiter = self.detect_delimiter(filepath)
                
                try:
                    data = np.loadtxt(filepath, delimiter=delimiter)
                except Exception:
                    # Se falhar, tenta sem delimitador (padrão numpy)
                    data = np.loadtxt(filepath)

                if data.ndim != 2 or data.shape[1] < 2:
                    raise ValueError(f"O arquivo {os.path.basename(filepath)} não parece ter duas colunas.")
                
                # Pega apenas as duas primeiras colunas (ignora outras)
                data = data[:, :2]
                
                filename = os.path.basename(filepath)
                if filename in self.loaded_data:
                    filename = f"{filename} ({len(self.loaded_data)})" # Evita duplicatas
                
                self.loaded_data[filename] = {
                    'wavelength': data[:, 0],
                    'intensity': data[:, 1]
                }
                self.file_listbox.insert('end', filename)

            if self.file_listbox.size() > 0:
                self.file_listbox.select_set(0)
                self.on_file_select(None)

        except Exception as e:
            messagebox.showerror("Erro ao Carregar Arquivo", f"Não foi possível ler os arquivos:\n{e}")
            self.reset_data()

    def on_file_select(self, event):
        """
        Chamado quando um item na Listbox é selecionado.
        Carrega os dados do arquivo selecionado para o estado "ativo".
        """
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            return
            
        selected_filename = self.file_listbox.get(selected_indices[0])
        
        if selected_filename in self.loaded_data:
            data = self.loaded_data[selected_filename]
            self.active_filename = selected_filename
            self.active_wavelength = data['wavelength']
            self.active_intensity = data['intensity']
            
            # Reseta estado
            self.active_filtered_intensity = None
            self.active_valley_wl = None
            self.active_valley_intensity = None
            self.save_full_spectrum_button.config(state='disabled')
            self.save_full_image_button.config(state='disabled')
            self.save_filtered_image_button.config(state='disabled')
            self.log_valley_button.config(state='disabled')
            self.valley_info_label.config(text="Vale do Espectro: N/A")
            
            # Plota o sinal original
            self.plot_data(self.active_wavelength, self.active_intensity, "Sinal Original")
            self.process_button.config(state='normal')

    def reset_data(self, clear_plot=True):
        """ Reseta o estado dos dados e da UI """
        self.loaded_data.clear()
        self.file_listbox.delete(0, 'end')
        self.active_wavelength = None
        self.active_intensity = None
        self.active_filename = None
        self.active_filtered_intensity = None
        self.active_valley_wl = None
        self.active_valley_intensity = None
        
        self.process_button.config(state='disabled')
        self.save_full_spectrum_button.config(state='disabled')
        self.save_full_image_button.config(state='disabled')
        self.save_filtered_image_button.config(state='disabled')
        self.log_valley_button.config(state='disabled')
        
        self.valley_info_label.config(text="Vale do Espectro: N/A")
        
        if clear_plot:
            self.ax.clear()
            self.ax.set_title("Carregue um arquivo para começar")
            self.ax.set_xlabel("Comprimento de Onda (nm)")
            self.ax.set_ylabel("Potência (dB)")
            self.ax.grid(True, linestyle=':', alpha=0.7)
            self.fig.tight_layout()
            self.canvas.draw()

    def process_and_plot(self, re_plot_only=False):
        """
        Valida os parâmetros, aplica o filtro, encontra o vale e atualiza o gráfico.
        Se re_plot_only=True, apenas redesenha com os dados atuais (ex: mudança de cor).
        """
        if self.active_wavelength is None or self.active_intensity is None:
            if not re_plot_only:
                messagebox.showwarning("Sem Dados", "Nenhum arquivo está selecionado.")
            return

        # --- Se não for apenas replot, calcula tudo ---
        if not re_plot_only:
            try:
                # --- Validação dos Parâmetros do Filtro ---
                try:
                    window_size = int(self.window_entry.get())
                    poly_order = int(self.order_entry.get())
                except ValueError:
                    messagebox.showerror("Erro de Parâmetro", "Janela e Ordem devem ser números inteiros.")
                    return

                if window_size % 2 == 0:
                    window_size += 1
                    self.window_entry.delete(0, 'end')
                    self.window_entry.insert(0, str(window_size))
                
                if poly_order >= window_size:
                    poly_order = max(1, window_size - 2)
                    self.order_entry.delete(0, 'end')
                    self.order_entry.insert(0, str(poly_order))
                elif poly_order < 1:
                     poly_order = 1
                     self.order_entry.delete(0, 'end')
                     self.order_entry.insert(0, str(poly_order))

                # --- Validação da Faixa de Busca ---
                try:
                    range_start = float(self.range_start_entry.get()) if self.range_start_entry.get() else min(self.active_wavelength)
                    range_end = float(self.range_end_entry.get()) if self.range_end_entry.get() else max(self.active_wavelength)
                except ValueError:
                    messagebox.showerror("Erro de Parâmetro", "Faixa de Busca deve ser numérica (ex: 1500.0)")
                    return
                
                if range_start >= range_end:
                    messagebox.showwarning("Aviso de Faixa", "O início da faixa deve ser menor que o fim.")
                    range_start, range_end = min(self.active_wavelength), max(self.active_wavelength)

                # --- Aplicação do Filtro ---
                data_to_filter = self.active_intensity.copy()
                if self.normalize_var.get():
                    data_to_filter = data_to_filter - np.max(data_to_filter)
                
                sinal_filtrado = savgol_filter(data_to_filter, window_size, poly_order)
                
                self.active_filtered_intensity = sinal_filtrado
                
                # --- Encontra o Vale (Dentro da Faixa) ---
                # 1. Cria uma máscara booleana para a faixa
                range_mask = (self.active_wavelength >= range_start) & (self.active_wavelength <= range_end)
                
                if not np.any(range_mask):
                    # Se a faixa for inválida (nenhum ponto dentro)
                    self.active_valley_wl = None
                    self.active_valley_intensity = None
                    info_text = "Vale do Espectro: Faixa inválida"
                else:
                    # 2. Aplica a máscara aos dados filtrados
                    wavelength_in_range = self.active_wavelength[range_mask]
                    intensity_in_range = sinal_filtrado[range_mask]
                    
                    # 3. Encontra o mínimo *apenas* dentro da faixa
                    min_intensity_index_local = np.argmin(intensity_in_range)
                    self.active_valley_intensity = intensity_in_range[min_intensity_index_local]
                    self.active_valley_wl = wavelength_in_range[min_intensity_index_local]
                    
                    info_text = f"Vale: {self.active_valley_intensity:.2f} dB @ {self.active_valley_wl:.2f} nm"
                
                # Atualiza UI
                self.valley_info_label.config(text=info_text)
                self.save_full_spectrum_button.config(state='normal')
                self.save_full_image_button.config(state='normal')
                self.save_filtered_image_button.config(state='normal')
                self.log_valley_button.config(state='normal' if self.active_valley_wl is not None else 'disabled')

            except Exception as e:
                messagebox.showerror("Erro no Filtro", f"Não foi possível aplicar o filtro:\n{e}")
                self.active_filtered_intensity = None
                self.active_valley_wl = None
                self.active_valley_intensity = None
                self.valley_info_label.config(text="Vale do Espectro: Erro")
                # Desabilita botões em caso de erro
                self.save_full_spectrum_button.config(state='disabled')
                self.save_full_image_button.config(state='disabled')
                self.save_filtered_image_button.config(state='disabled')
                self.log_valley_button.config(state='disabled')
                return # Não plota se deu erro

        # --- (Re)Plotagem ---
        # Prepara os dados originais para plotar (normalizados ou não)
        original_plot_data = self.active_intensity
        if self.normalize_var.get() and self.active_filtered_intensity is not None:
            # Se normalizou, plota o original normalizado para bater com o filtro
            original_plot_data = self.active_intensity - np.max(self.active_intensity)
            
        original_label = "Sinal Original"
        if self.normalize_var.get():
            original_label += " (Normalizado)"
            
        filtered_label = "Sinal Filtrado"
        if self.normalize_var.get():
            filtered_label += " (Normalizado)"

        # Chama a função de plotagem
        self.plot_data(
            w_orig=self.active_wavelength,
            i_orig=original_plot_data,
            label_orig=original_label,
            w_filt=self.active_wavelength if self.active_filtered_intensity is not None else None,
            i_filt=self.active_filtered_intensity,
            label_filt=filtered_label
        )


    def plot_data(self, w_orig, i_orig, label_orig, w_filt=None, i_filt=None, label_filt=None):
        """
        Limpa e redesenha o gráfico do Matplotlib.
        """
        # Salva o zoom atual
        if not self.ax.get_title().startswith("Carregue"): # Não salva o zoom inicial
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
        else:
            xlim = None
            ylim = None

        self.ax.clear()
        
        # Plot Original
        self.ax.plot(w_orig, i_orig, '.', color=self.color_original, markersize=2, label=label_orig)
        
        # Plot Filtrado (se fornecido)
        if w_filt is not None and i_filt is not None and label_filt is not None:
            self.ax.plot(w_filt, i_filt, '-', color=self.color_filtrado, linewidth=2, label=label_filt)

            # Anota o vale no gráfico (se encontrado)
            if self.active_valley_wl is not None:
                wv_min = self.active_valley_wl
                int_min = self.active_valley_intensity
                text_label = f"Vale: {int_min:.2f} dB\n@ {wv_min:.2f} nm"
                
                # Tenta desenhar a anotação
                try:
                    self.ax.annotate(text_label,
                        xy=(wv_min, int_min),
                        xytext=(wv_min + (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 0.05, int_min + abs(int_min)*0.1),
                        ha='left',
                        va='bottom',
                        arrowprops=dict(arrowstyle='->', color=self.color_filtrado, connectionstyle='arc3,rad=0.3'),
                        bbox=dict(boxstyle='round,pad=0.3', fc=self.color_filtrado, alpha=0.2),
                        color='black' # Cor do texto
                    )
                except Exception as e:
                    print(f"Erro ao desenhar anotação: {e}") # Não trava o app

            # Desenha a Faixa de Busca (v7)
            try:
                range_start = float(self.range_start_entry.get()) if self.range_start_entry.get() else None
                range_end = float(self.range_end_entry.get()) if self.range_end_entry.get() else None
                
                plot_ymin, plot_ymax = self.ax.get_ylim()
                if range_start:
                    self.ax.vlines(range_start, plot_ymin, plot_ymax, colors='blue', linestyles='dashed', alpha=0.5)
                if range_end:
                    self.ax.vlines(range_end, plot_ymin, plot_ymax, colors='blue', linestyles='dashed', alpha=0.5)
            except Exception:
                pass # Ignora erros de faixa se não forem numéricos

        self.ax.set_title(f"Espectro de: {self.active_filename}")
        self.ax.set_xlabel("Comprimento de Onda (nm)")
        self.ax.set_ylabel("Potência (dB)")
        self.ax.legend()
        self.ax.grid(True, linestyle=':', alpha=0.7)
        
        # Restaura o zoom/pan anterior
        if xlim and ylim:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
        
        self.fig.tight_layout()
        self.canvas.draw()

    # ===================================================================
    # FUNÇÕES DE SALVAMENTO
    # ===================================================================

    def _ask_save_filepath(self, title, initial_filename):
        """Helper para perguntar onde salvar (XLSX ou CSV)."""
        return filedialog.asksaveasfilename(
            title=title,
            initialfile=initial_filename,
            filetypes=[
                ("Arquivo Excel", "*.xlsx"),
                ("Arquivo CSV (separado por ;)", "*.csv"),
                ("Todos os arquivos", "*.*")
            ]
        )

    def _write_to_file(self, df, filepath):
        """Helper para escrever DataFrame em XLSX ou CSV."""
        if filepath.endswith('.xlsx'):
            df.to_excel(filepath, index=False, sheet_name='Dados')
        elif filepath.endswith('.csv'):
            df.to_csv(filepath, index=False, sep=';', decimal='.') # Ponto decimal é mais robusto
        else:
            raise ValueError(f"Extensão de arquivo não suportada: {filepath}")

    def save_full_spectrum(self):
        """Salva o espectro COMPLETO (original e filtrado) em .xlsx ou .csv."""
        if self.active_filtered_intensity is None or self.active_wavelength is None:
            messagebox.showwarning("Sem Dados", "Nenhum dado filtrado para salvar.")
            return

        suggested_filename = f"{os.path.splitext(self.active_filename)[0]}_espectro_completo.xlsx"
        filepath = self._ask_save_filepath("Salvar espectro completo", suggested_filename)
        if not filepath:
            return

        try:
            data_to_save = {
                'Comprimento de Onda (nm)': self.active_wavelength,
                'Intensidade Original (dB)': self.active_intensity,
                'Intensidade Filtrada (dB)': self.active_filtered_intensity
            }
            df = pd.DataFrame(data_to_save)
            self._write_to_file(df, filepath)
            messagebox.showinfo("Sucesso", f"Espectro completo salvo em:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Erro ao Salvar", f"Não foi possível salvar o arquivo:\n{e}")

    def set_log_file(self):
        """Pede ao usuário para definir um arquivo .xlsx/.csv que servirá de log."""
        filepath = self._ask_save_filepath("Definir arquivo de Log de Vales", "log_vales_lpg.xlsx")
        if filepath:
            self.log_filepath = filepath
            # Mostra o nome do arquivo, não o caminho completo, se for muito longo
            display_path = os.path.basename(filepath)
            if len(self.log_filepath) > 40:
                display_path = f"...{self.log_filepath[-40:]}"
            self.log_file_label.config(text=display_path, fg="black")
            # Tooltip com o caminho completo
            # (Implementação simples de tooltip, poderia ser melhorada com Ttk)
            # self.log_file_label.bind("<Enter>", lambda e: self.show_tooltip(self.log_filepath))
            # self.log_file_label.bind("<Leave>", lambda e: self.hide_tooltip())

    def log_valley_data(self):
        """Adiciona o vale atual como uma nova linha no arquivo de log."""
        if self.active_valley_wl is None:
            messagebox.showwarning("Sem Dados", "Nenhum vale detectado.")
            return
            
        if not self.log_filepath:
            messagebox.showwarning("Sem Log", "Defina um arquivo de Log primeiro.")
            self.set_log_file()
            if not self.log_filepath:
                return
                
        sample_name = self.sample_name_entry.get()
        if not sample_name:
            messagebox.showwarning("Sem Amostra", "Por favor, insira um nome para a amostra.")
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        new_data_row = {
            'horario': [timestamp],
            'comprimento_onda_filtrado (nm)': [self.active_valley_wl],
            'intensidade_filtrada_vale (dB)': [self.active_valley_intensity],
            'amostra': [sample_name],
            'arquivo_origem': [self.active_filename]
        }
        df_new_row = pd.DataFrame(new_data_row)

        try:
            df_log = None
            if os.path.exists(self.log_filepath):
                # Tenta ler o arquivo existente
                if self.log_filepath.endswith('.xlsx'):
                    df_log = pd.read_excel(self.log_filepath)
                else:
                    df_log = pd.read_csv(self.log_filepath, sep=';')
                
                df_log = pd.concat([df_log, df_new_row], ignore_index=True)
            else:
                # Arquivo não existe, este é o primeiro registro
                df_log = df_new_row
            
            self._write_to_file(df_log, self.log_filepath)
            messagebox.showinfo("Sucesso", f"Vale registrado com sucesso em:\n{os.path.basename(self.log_filepath)}")
        
        except PermissionError:
             messagebox.showerror("Erro de Permissão", f"Não foi possível salvar o log.\nO arquivo '{os.path.basename(self.log_filepath)}' está aberto?\n\nFeche-o e tente novamente.")
        except Exception as e:
            messagebox.showerror("Erro ao Salvar Log", f"Ocorreu um erro inesperado:\n{e}")

    def save_plot_image(self):
        """Salva a imagem do gráfico COMPLETO usando o diálogo do Matplotlib."""
        try:
            # O diálogo de salvar do Matplotlib é chamado diretamente
            self.canvas.toolbar.save_figure()
        except Exception as e:
            messagebox.showerror("Erro ao Salvar Imagem", f"Ocorreu um erro: {e}")

    def save_filtered_plot_only(self):
        """Salva uma imagem contendo APENAS o gráfico filtrado, respeitando o zoom e opções."""
        if self.active_filtered_intensity is None:
            messagebox.showwarning("Sem Dados", "Nenhum dado filtrado para salvar.")
            return

        suggested_filename = f"{os.path.splitext(self.active_filename)[0]}_grafico_filtrado.png"
        filepath = filedialog.asksaveasfilename(
            title="Salvar Imagem Apenas do Filtro",
            initialfile=suggested_filename,
            filetypes=[("Imagem PNG", "*.png"), ("Imagem PDF", "*.pdf"), ("Imagem SVG", "*.svg"), ("Todos os arquivos", "*.*")]
        )
        if not filepath:
            return

        # --- Cria uma nova figura temporária ---
        try:
            fig_temp, ax_temp = plt.subplots(figsize=(10, 6)) # Tamanho padrão de boa qualidade
            
            # 1. Plotar o filtro
            ax_temp.plot(self.active_wavelength, self.active_filtered_intensity, '-', 
                         color=self.color_filtrado, linewidth=2, label="Sinal Filtrado")

            # 2. Incluir anotação (se marcada)
            if self.include_annotation_var.get() and self.active_valley_wl is not None:
                wv_min = self.active_valley_wl
                int_min = self.active_valley_intensity
                text_label = f"Vale: {int_min:.2f} dB\n@ {wv_min:.2f} nm"
                
                # Obtém os limites para posicionar a seta
                xlim_temp = ax_temp.get_xlim()
                
                ax_temp.annotate(text_label,
                    xy=(wv_min, int_min),
                    xytext=(wv_min + (xlim_temp[1] - xlim_temp[0]) * 0.05, int_min + abs(int_min)*0.1),
                    ha='left', va='bottom',
                    arrowprops=dict(arrowstyle='->', color=self.color_filtrado, connectionstyle='arc3,rad=0.3'),
                    bbox=dict(boxstyle='round,pad=0.3', fc=self.color_filtrado, alpha=0.2)
                )

            # 3. Incluir faixa de busca (se marcada)
            if self.include_range_var.get():
                try:
                    range_start = float(self.range_start_entry.get()) if self.range_start_entry.get() else None
                    range_end = float(self.range_end_entry.get()) if self.range_end_entry.get() else None
                    
                    plot_ymin, plot_ymax = ax_temp.get_ylim()
                    if range_start:
                        ax_temp.vlines(range_start, plot_ymin, plot_ymax, colors='blue', linestyles='dashed', alpha=0.5)
                    if range_end:
                        ax_temp.vlines(range_end, plot_ymin, plot_ymax, colors='blue', linestyles='dashed', alpha=0.5)
                except Exception:
                    pass # Ignora

            # 4. Configurar eixos e labels
            ax_temp.set_title(f"Espectro Filtrado de: {self.active_filename}")
            ax_temp.set_xlabel("Comprimento de Onda (nm)")
            ax_temp.set_ylabel("Potência (dB)")
            ax_temp.legend()
            ax_temp.grid(True, linestyle=':', alpha=0.7)
            
            # 5. Aplicar o Zoom/Pan do gráfico principal
            main_xlim = self.ax.get_xlim()
            main_ylim = self.ax.get_ylim()
            ax_temp.set_xlim(main_xlim)
            ax_temp.set_ylim(main_ylim)

            # 6. Salvar
            fig_temp.tight_layout()
            fig_temp.savefig(filepath, dpi=300)
            
            # 7. Fechar a figura temporária para liberar memória
            plt.close(fig_temp)
            
            messagebox.showinfo("Sucesso", f"Imagem (apenas filtro) salva em:\n{filepath}")

        except Exception as e:
            messagebox.showerror("Erro ao Salvar Imagem", f"Não foi possível salvar a imagem:\n{e}")
            if 'fig_temp' in locals():
                plt.close(fig_temp) # Garante que fecha em caso de erro


if __name__ == "__main__":
    root = tk.Tk()
    app = LpgFilterApp(root)
    root.mainloop()