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
        v13.0: Adiciona processamento em lote, barra de progresso e separador de "Análise Temporal".
        """
        self.master = master
        master.title("Filtro Savitzky-Golay (v13.0 - Processamento em Lote)")
        master.geometry("1200x800")

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
        
        # v13: Para guardar os resultados do lote
        self.last_batch_results = None 

        # --- Estrutura Principal (Grid) ---
        main_frame = tk.Frame(master)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=0) # Coluna de controles
        main_frame.grid_columnconfigure(1, weight=1) # Coluna do gráfico

        # --- Coluna da Esquerda: Controles (com Scrollbar v12.0) ---
        control_canvas_frame = tk.Frame(main_frame)
        control_canvas_frame.grid(row=0, column=0, sticky="ns", padx=(0, 5))
        control_canvas_frame.grid_rowconfigure(0, weight=1)
        
        control_canvas = tk.Canvas(control_canvas_frame, width=350, borderwidth=0, highlightthickness=0)
        control_scrollbar = ttk.Scrollbar(control_canvas_frame, orient="vertical", command=control_canvas.yview)
        control_canvas.configure(yscrollcommand=control_scrollbar.set)
        
        control_scrollbar.grid(row=0, column=1, sticky="ns")
        control_canvas.grid(row=0, column=0, sticky="ns")
        
        self.control_frame = tk.Frame(control_canvas)
        control_canvas.create_window((0, 0), window=self.control_frame, anchor="nw")

        def update_scrollregion(event):
            control_canvas.configure(scrollregion=control_canvas.bbox("all"))
        
        self.control_frame.bind("<Configure>", update_scrollregion)
        
        def on_mousewheel(event):
            # Normalização do delta
            delta = 0
            if event.num == 4: delta = -1 # Linux scroll up
            elif event.num == 5: delta = 1  # Linux scroll down
            elif hasattr(event, 'delta'): delta = -1 * (event.delta // 120) # Windows/Mac
            
            control_canvas.yview_scroll(delta, "units")

        # Bind em 'control_canvas' e 'self.control_frame' pode ser mais robusto
        self.master.bind_all("<MouseWheel>", on_mousewheel, add='+')
        self.master.bind_all("<Button-4>", on_mousewheel, add='+')
        self.master.bind_all("<Button-5>", on_mousewheel, add='+')


        # --- Botão Carregar (dentro de self.control_frame) ---
        self.load_button = tk.Button(self.control_frame, text="Carregar Arquivo(s) (.txt)", command=self.load_files)
        self.load_button.pack(fill='x', pady=(5, 10), padx=5)

        # --- Frame: Customização do Gráfico (v10) ---
        color_frame = tk.LabelFrame(self.control_frame, text="Customização do Gráfico")
        color_frame.pack(fill='x', pady=5, padx=5)
        
        self.color_orig_button = tk.Button(color_frame, text="Cor Original", command=lambda: self.pick_color('original'))
        self.color_orig_button.pack(side='left', padx=5, expand=True, fill='x', pady=5)
        self.color_preview_orig = tk.Label(color_frame, text="", bg=self.color_original, relief='sunken', width=4)
        self.color_preview_orig.pack(side='left', padx=(0, 10))

        self.color_filt_button = tk.Button(color_frame, text="Cor Filtrada", command=lambda: self.pick_color('filtrado'))
        self.color_filt_button.pack(side='left', padx=5, expand=True, fill='x', pady=5)
        self.color_preview_filt = tk.Label(color_frame, text="", bg=self.color_filtrado, relief='sunken', width=4)
        self.color_preview_filt.pack(side='left', padx=(0, 5))

        # --- Frame: Controles de Filtro e Busca (v7) ---
        filter_frame = tk.LabelFrame(self.control_frame, text="Controles de Filtro e Busca")
        filter_frame.pack(fill='x', pady=5, padx=5)

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

        self.process_button = tk.Button(filter_frame, text="Aplicar Filtro (Ficheiro Único)", command=self.process_and_plot, state='disabled')
        self.process_button.pack(fill='x', padx=5, pady=(5, 10))

        # --- Frame: Log de Vales (Resultados) (v6) ---
        log_frame = tk.LabelFrame(self.control_frame, text="Log de Vales (Resultados)")
        log_frame.pack(fill='x', pady=5, padx=5)

        tk.Label(log_frame, text="Nome da Amostra (Base p/ Lote):").pack(anchor='w', padx=5)
        self.sample_name_entry = tk.Entry(log_frame)
        self.sample_name_entry.pack(fill='x', padx=5, pady=(0, 5))

        self.set_log_button = tk.Button(log_frame, text="Definir Arquivo de Log (.xlsx/.csv)", command=self.set_log_file)
        self.set_log_button.pack(fill='x', padx=5, pady=5)

        self.log_valley_button = tk.Button(log_frame, text="Registrar Vale Único no Log", command=self.log_single_valley_data, state='disabled')
        self.log_valley_button.pack(fill='x', padx=5, pady=5)
        
        # --- NOVO (v13): Botão de Lote ---
        self.batch_process_button = tk.Button(log_frame, text="Analisar e Registrar LOTE no Log", 
                                              command=self.batch_process_and_log, state='disabled',
                                              font=("Helvetica", 10, "bold"), bg="#D0E8D0")
        self.batch_process_button.pack(fill='x', padx=5, pady=5)

        tk.Label(log_frame, text="Arquivo de Log:", anchor='w').pack(fill='x', padx=5, pady=(5,0))
        self.log_file_label = tk.Label(log_frame, text="Nenhum definido", fg="gray", anchor='w', justify='left', relief='sunken', borderwidth=1)
        self.log_file_label.pack(fill='x', padx=5, pady=(0, 5))

        # --- NOVO (v13): Barra de Progresso ---
        progress_frame = tk.LabelFrame(self.control_frame, text="Progresso do Lote")
        progress_frame.pack(fill='x', pady=5, padx=5)
        self.progress_bar = ttk.Progressbar(progress_frame, orient='horizontal', mode='determinate')
        self.progress_bar.pack(fill='x', padx=5, pady=5)
        self.progress_label = tk.Label(progress_frame, text="Aguardando lote...", anchor='w')
        self.progress_label.pack(fill='x', padx=5, pady=(0,5))


        # --- Frame: Arquivos de Espectro (v3) ---
        list_frame = tk.LabelFrame(self.control_frame, text="Arquivos de Espectro")
        list_frame.pack(fill='x', pady=5, padx=5, expand=False)
        
        self.valley_info_label = tk.Label(list_frame, text="Vale do Espectro: N/A", fg="blue", font=("Helvetica", 10, "bold"))
        self.valley_info_label.pack(anchor='w', padx=5, pady=2)
        
        list_subframe = tk.Frame(list_frame, height=150)
        list_subframe.pack(fill='x', expand=False)
        list_subframe.pack_propagate(False)

        self.scrollbar = tk.Scrollbar(list_subframe, orient='vertical')
        self.file_listbox = tk.Listbox(list_subframe, yscrollcommand=self.scrollbar.set, exportselection=False)
        self.scrollbar.config(command=self.file_listbox.yview)
        self.scrollbar.pack(side='right', fill='y')
        self.file_listbox.pack(side='left', fill='both', expand=True)
        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_select)
        
        # --- Frame: Salvamento ---
        save_main_frame = tk.LabelFrame(self.control_frame, text="Opções de Salvamento (Ficheiro Único)")
        save_main_frame.pack(fill='x', pady=(10, 0), padx=5)

        self.save_full_spectrum_button = tk.Button(save_main_frame, text="Salvar Espectro Completo (.xlsx/.csv)", command=self.save_full_spectrum, state='disabled')
        self.save_full_spectrum_button.pack(fill='x', padx=5, pady=5)

        self.save_full_image_button = tk.Button(save_main_frame, text="Salvar Imagem (Completa)", command=self.save_plot_image, state='disabled')
        self.save_full_image_button.pack(fill='x', padx=5, pady=5)

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


        # --- Coluna da Direita: Gráfico (COM SEPARADORES v13) ---
        self.graph_frame = tk.Frame(main_frame, bg='white')
        self.graph_frame.grid(row=0, column=1, sticky="nsew")
        self.graph_frame.grid_rowconfigure(0, weight=1)
        self.graph_frame.grid_columnconfigure(0, weight=1)
        
        self.notebook = ttk.Notebook(self.graph_frame)
        self.notebook.pack(fill='both', expand=True)
        
        # --- Separador 1: Espectro Atual ---
        spectrum_tab = tk.Frame(self.notebook, bg='white')
        self.notebook.add(spectrum_tab, text='Espectro Atual')
        
        spectrum_tab.grid_rowconfigure(0, weight=1)
        spectrum_tab.grid_columnconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=spectrum_tab)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, spectrum_tab, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.grid(row=1, column=0, sticky="ew")
        
        self.ax.set_title("Carregue um ou mais arquivos para começar")
        self.ax.set_xlabel("Comprimento de Onda (nm)")
        self.ax.set_ylabel("Potência (dB)")
        self.ax.grid(True, linestyle=':', alpha=0.7)
        self.fig.tight_layout()
        
        # --- Separador 2: Análise Temporal ---
        self.time_series_tab = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.time_series_tab, text='Análise Temporal', state='disabled')
        
        self.time_series_tab.grid_rowconfigure(0, weight=1)
        self.time_series_tab.grid_columnconfigure(0, weight=1)

        self.ts_fig, self.ts_ax = plt.subplots()
        self.ts_canvas = FigureCanvasTkAgg(self.ts_fig, master=self.time_series_tab)
        self.ts_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        self.ts_toolbar = NavigationToolbar2Tk(self.ts_canvas, self.time_series_tab, pack_toolbar=False)
        self.ts_toolbar.update()
        self.ts_toolbar.grid(row=1, column=0, sticky="ew")
        
        self.ts_ax.set_title("Processe um lote para ver a análise temporal")
        self.ts_ax.set_xlabel("Índice do Arquivo (Tempo)")
        self.ts_ax.set_ylabel("Comprimento de Onda do Vale (nm)")
        self.ts_ax.grid(True, linestyle=':', alpha=0.7)
        self.ts_fig.tight_layout()

    # ===================================================================
    # FUNÇÕES DE LÓGICA
    # ===================================================================

    def pick_color(self, target):
        color_code = colorchooser.askcolor(title=f"Escolher cor para {target}")
        if color_code[1]:
            if target == 'original':
                self.color_original = color_code[1]
                self.color_preview_orig.config(bg=self.color_original)
            elif target == 'filtrado':
                self.color_filtrado = color_code[1]
                self.color_preview_filt.config(bg=self.color_filtrado)
            
            if self.active_wavelength is not None:
                self.process_and_plot(re_plot_only=True)


    def detect_delimiter(self, filepath):
        try:
            with open(filepath, 'r') as f:
                first_line = f.readline()
            if ';' in first_line: return ';'
            if re.search(r'\d,\d', first_line): return None
            if ',' in first_line: return ','
            return None
        except Exception: return None

    def load_files(self):
        filepaths = filedialog.askopenfilenames(
            title="Selecione o(s) arquivo(s) de espectro",
            filetypes=(("Arquivos de Texto", "*.txt"), ("Todos os arquivos", "*.*"))
        )
        if not filepaths: return

        self.reset_data(clear_plot=False)
        
        try:
            for filepath in filepaths:
                delimiter = self.detect_delimiter(filepath)
                try: data = np.loadtxt(filepath, delimiter=delimiter)
                except Exception: data = np.loadtxt(filepath)

                if data.ndim != 2 or data.shape[1] < 2:
                    raise ValueError(f"O arquivo {os.path.basename(filepath)} não parece ter duas colunas.")
                
                data = data[:, :2]
                filename = os.path.basename(filepath)
                if filename in self.loaded_data:
                    filename = f"{filename}_({len(self.loaded_data)})"
                
                self.loaded_data[filename] = {'wavelength': data[:, 0], 'intensity': data[:, 1]}
                self.file_listbox.insert('end', filename)

            if self.file_listbox.size() > 0:
                self.file_listbox.select_set(0)
                self.on_file_select(None)
                self.batch_process_button.config(state='normal') # Ativa o botão de lote

        except Exception as e:
            messagebox.showerror("Erro ao Carregar Arquivo", f"Não foi possível ler os arquivos:\n{e}")
            self.reset_data()

    def on_file_select(self, event):
        selected_indices = self.file_listbox.curselection()
        if not selected_indices: return
            
        selected_filename = self.file_listbox.get(selected_indices[0])
        
        if selected_filename in self.loaded_data:
            data = self.loaded_data[selected_filename]
            self.active_filename = selected_filename
            self.active_wavelength = data['wavelength']
            self.active_intensity = data['intensity']
            
            self.active_filtered_intensity = None
            self.active_valley_wl = None
            self.active_valley_intensity = None
            self.save_full_spectrum_button.config(state='disabled')
            self.save_full_image_button.config(state='disabled')
            self.save_filtered_image_button.config(state='disabled')
            self.log_valley_button.config(state='disabled')
            self.valley_info_label.config(text="Vale do Espectro: N/A")
            
            self.plot_data(self.active_wavelength, self.active_intensity, "Sinal Original")
            self.process_button.config(state='normal')
            self.notebook.select(0) # Volta para o separador do espectro

    def reset_data(self, clear_plot=True):
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
        self.batch_process_button.config(state='disabled')
        
        self.valley_info_label.config(text="Vale do Espectro: N/A")
        
        if clear_plot:
            self.ax.clear()
            self.ax.set_title("Carregue um arquivo para começar")
            self.ax.set_xlabel("Comprimento de Onda (nm)")
            self.ax.set_ylabel("Potência (dB)")
            self.ax.grid(True, linestyle=':', alpha=0.7)
            self.fig.tight_layout()
            self.canvas.draw()
            
            self.ts_ax.clear()
            self.ts_ax.set_title("Processe um lote para ver a análise temporal")
            self.ts_ax.set_xlabel("Índice do Arquivo (Tempo)")
            self.ts_ax.set_ylabel("Comprimento de Onda do Vale (nm)")
            self.ts_ax.grid(True, linestyle=':', alpha=0.7)
            self.ts_fig.tight_layout()
            self.ts_canvas.draw()
            self.notebook.add(self.time_series_tab, state='disabled')

    def _get_filter_params(self):
        """Helper para validar e buscar parâmetros de filtro e faixa."""
        try:
            window_size = int(self.window_entry.get())
            poly_order = int(self.order_entry.get())
        except ValueError:
            messagebox.showerror("Erro de Parâmetro", "Janela e Ordem devem ser números inteiros.")
            return None

        if window_size % 2 == 0:
            window_size += 1
            self.window_entry.delete(0, 'end'); self.window_entry.insert(0, str(window_size))
        
        if poly_order >= window_size:
            poly_order = max(1, window_size - 2)
            self.order_entry.delete(0, 'end'); self.order_entry.insert(0, str(poly_order))
        elif poly_order < 1:
             poly_order = 1
             self.order_entry.delete(0, 'end'); self.order_entry.insert(0, str(poly_order))

        try:
            # Usa min/max dos *dados ativos* como fallback se os campos estiverem vazios
            wl_min = min(self.active_wavelength) if self.active_wavelength is not None else 0
            wl_max = max(self.active_wavelength) if self.active_wavelength is not None else 1
            
            range_start = float(self.range_start_entry.get()) if self.range_start_entry.get() else wl_min
            range_end = float(self.range_end_entry.get()) if self.range_end_entry.get() else wl_max
        except ValueError:
            messagebox.showerror("Erro de Parâmetro", "Faixa de Busca deve ser numérica (ex: 1500.0)")
            return None
        
        if range_start >= range_end:
            messagebox.showwarning("Aviso de Faixa", "O início da faixa deve ser menor que o fim.")
            range_start, range_end = wl_min, wl_max
            
        return window_size, poly_order, range_start, range_end, self.normalize_var.get()

    def _find_valley(self, wavelengths, intensities, range_start, range_end):
        """Helper para encontrar o vale dentro de uma faixa."""
        range_mask = (wavelengths >= range_start) & (wavelengths <= range_end)
        
        if not np.any(range_mask):
            return None # Faixa inválida
        
        wavelength_in_range = wavelengths[range_mask]
        intensity_in_range = intensities[range_mask]
        
        min_intensity_index_local = np.argmin(intensity_in_range)
        valley_intensity = intensity_in_range[min_intensity_index_local]
        valley_wl = wavelength_in_range[min_intensity_index_local]
        
        return valley_wl, valley_intensity

    def process_and_plot(self, re_plot_only=False):
        if self.active_wavelength is None or self.active_intensity is None:
            if not re_plot_only: messagebox.showwarning("Sem Dados", "Nenhum arquivo está selecionado.")
            return

        if not re_plot_only:
            params = self._get_filter_params()
            if params is None: return # Erro na validação
            
            window_size, poly_order, range_start, range_end, normalize = params

            try:
                data_to_filter = self.active_intensity.copy()
                if normalize:
                    data_to_filter = data_to_filter - np.max(data_to_filter)
                
                sinal_filtrado = savgol_filter(data_to_filter, window_size, poly_order)
                self.active_filtered_intensity = sinal_filtrado
                
                # --- Encontra o Vale (Dentro da Faixa) ---
                valley_result = self._find_valley(self.active_wavelength, sinal_filtrado, range_start, range_end)
                
                if valley_result is None:
                    self.active_valley_wl = None
                    self.active_valley_intensity = None
                    info_text = "Vale do Espectro: Faixa inválida"
                else:
                    self.active_valley_wl, self.active_valley_intensity = valley_result
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
                self.save_full_spectrum_button.config(state='disabled')
                self.save_full_image_button.config(state='disabled')
                self.save_filtered_image_button.config(state='disabled')
                self.log_valley_button.config(state='disabled')
                return

        # --- (Re)Plotagem ---
        original_plot_data = self.active_intensity
        if self.normalize_var.get() and self.active_filtered_intensity is not None:
            original_plot_data = self.active_intensity - np.max(self.active_intensity)
            
        original_label = "Sinal Original" + (" (Normalizado)" if self.normalize_var.get() else "")
        filtered_label = "Sinal Filtrado" + (" (Normalizado)" if self.normalize_var.get() else "")

        self.plot_data(
            w_orig=self.active_wavelength,
            i_orig=original_plot_data,
            label_orig=original_label,
            w_filt=self.active_wavelength if self.active_filtered_intensity is not None else None,
            i_filt=self.active_filtered_intensity,
            label_filt=filtered_label
        )


    def plot_data(self, w_orig, i_orig, label_orig, w_filt=None, i_filt=None, label_filt=None):
        if not self.ax.get_title().startswith("Carregue"):
            xlim = self.ax.get_xlim(); ylim = self.ax.get_ylim()
        else: xlim = None; ylim = None

        self.ax.clear()
        self.ax.plot(w_orig, i_orig, '.', color=self.color_original, markersize=2, label=label_orig)
        
        if w_filt is not None and i_filt is not None and label_filt is not None:
            self.ax.plot(w_filt, i_filt, '-', color=self.color_filtrado, linewidth=2, label=label_filt)

            if self.active_valley_wl is not None:
                wv_min = self.active_valley_wl
                int_min = self.active_valley_intensity
                text_label = f"Vale: {int_min:.2f} dB\n@ {wv_min:.2f} nm"
                
                try:
                    self.ax.annotate(text_label,
                        xy=(wv_min, int_min),
                        xytext=(wv_min + (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 0.05, int_min + abs(int_min)*0.1),
                        ha='left', va='bottom',
                        arrowprops=dict(arrowstyle='->', color=self.color_filtrado, connectionstyle='arc3,rad=0.3'),
                        bbox=dict(boxstyle='round,pad=0.3', fc=self.color_filtrado, alpha=0.2),
                        color='black'
                    )
                except Exception: pass # Ignora erro de anotação

            try:
                range_start = float(self.range_start_entry.get()) if self.range_start_entry.get() else None
                range_end = float(self.range_end_entry.get()) if self.range_end_entry.get() else None
                plot_ymin, plot_ymax = self.ax.get_ylim()
                if range_start: self.ax.vlines(range_start, plot_ymin, plot_ymax, colors='blue', linestyles='dashed', alpha=0.5)
                if range_end: self.ax.vlines(range_end, plot_ymin, plot_ymax, colors='blue', linestyles='dashed', alpha=0.5)
            except Exception: pass

        self.ax.set_title(f"Espectro de: {self.active_filename}")
        self.ax.set_xlabel("Comprimento de Onda (nm)")
        self.ax.set_ylabel("Potência (dB)")
        self.ax.legend()
        self.ax.grid(True, linestyle=':', alpha=0.7)
        
        if xlim and ylim: self.ax.set_xlim(xlim); self.ax.set_ylim(ylim)
        
        self.fig.tight_layout()
        self.canvas.draw()
        
    def _plot_time_series(self, ts_data):
        """Plota os dados da análise temporal no separador 'Análise Temporal'."""
        self.ts_ax.clear()
        
        if not ts_data:
            self.ts_ax.set_title("Nenhum dado válido encontrado no lote.")
            self.ts_canvas.draw()
            return

        indices = [d[0] for d in ts_data]
        wavelengths = [d[1] for d in ts_data]
        intensities = [d[2] for d in ts_data]
        
        # Plot do Comprimento de Onda
        self.ts_ax.plot(indices, wavelengths, 'o-', color='blue', label='Comprimento de Onda (nm)')
        self.ts_ax.set_xlabel("Índice do Arquivo (Tempo)")
        self.ts_ax.set_ylabel("Comprimento de Onda do Vale (nm)", color='blue')
        self.ts_ax.tick_params(axis='y', labelcolor='blue')
        
        # Cria um segundo eixo Y para a Intensidade
        ax2 = self.ts_ax.twinx()
        ax2.plot(indices, intensities, 's--', color='red', alpha=0.6, label='Intensidade (dB)')
        ax2.set_ylabel("Intensidade do Vale (dB)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        self.ts_ax.set_title(f"Análise Temporal de {len(indices)} Pontos")
        self.ts_ax.grid(True, linestyle=':', alpha=0.7)
        self.ts_fig.tight_layout()
        self.ts_canvas.draw()
        
        # Ativa e seleciona o separador
        self.notebook.add(self.time_series_tab, state='normal')
        self.notebook.select(1)


    # ===================================================================
    # FUNÇÕES DE SALVAMENTO (v13)
    # ===================================================================

    def _ask_save_filepath(self, title, initial_filename):
        # Pega a extensão padrão do nome inicial
        default_ext = os.path.splitext(initial_filename)[1]
        
        return filedialog.asksaveasfilename(
            title=title, 
            initialfile=initial_filename,
            # ADICIONA defaultextension para forçar a extensão se o usuário esquecer
            defaultextension=default_ext,
            filetypes=[("Arquivo Excel", "*.xlsx"), ("Arquivo CSV (separado por ;)", "*.csv"), ("Todos os arquivos", "*.*")]
        )

    def _write_to_file(self, df, filepath):
        try:
            if filepath.endswith('.xlsx'):
                df.to_excel(filepath, index=False, sheet_name='Dados')
            elif filepath.endswith('.csv'):
                df.to_csv(filepath, index=False, sep=';', decimal='.')
            else:
                raise ValueError(f"Extensão de arquivo não suportada: {filepath}")
            return True
        except PermissionError:
             messagebox.showerror("Erro de Permissão", f"Não foi possível salvar.\nO arquivo '{os.path.basename(filepath)}' está aberto?\n\nFeche-o e tente novamente.")
             return False
        except Exception as e:
            messagebox.showerror("Erro ao Salvar", f"Não foi possível salvar o arquivo:\n{e}")
            return False

    def _append_to_log(self, df_to_append):
        """Helper central para ler, adicionar e salvar no arquivo de log."""
        if not self.log_filepath:
            messagebox.showwarning("Sem Log", "Defina um arquivo de Log primeiro.")
            self.set_log_file()
            if not self.log_filepath:
                return False
        
        try:
            df_log = None
            if os.path.exists(self.log_filepath):
                if self.log_filepath.endswith('.xlsx'):
                    df_log = pd.read_excel(self.log_filepath)
                else:
                    df_log = pd.read_csv(self.log_filepath, sep=';')
                
                df_log = pd.concat([df_log, df_to_append], ignore_index=True)
            else:
                df_log = df_to_append
            
            return self._write_to_file(df_log, self.log_filepath)
        
        except Exception as e:
            messagebox.showerror("Erro ao Salvar Log", f"Ocorreu um erro inesperado ao aceder ao log:\n{e}")
            return False

    def save_full_spectrum(self):
        if self.active_filtered_intensity is None or self.active_wavelength is None:
            messagebox.showwarning("Sem Dados", "Nenhum dado filtrado para salvar.")
            return

        suggested_filename = f"{os.path.splitext(self.active_filename)[0]}_espectro_completo.xlsx"
        filepath = self._ask_save_filepath("Salvar espectro completo", suggested_filename)
        if not filepath: return

        data_to_save = {
            'Comprimento de Onda (nm)': self.active_wavelength,
            'Intensidade Original (dB)': self.active_intensity,
            'Intensidade Filtrada (dB)': self.active_filtered_intensity
        }
        df = pd.DataFrame(data_to_save)
        if self._write_to_file(df, filepath):
            messagebox.showinfo("Sucesso", f"Espectro completo salvo em:\n{filepath}")

    def set_log_file(self):
        filepath = self._ask_save_filepath("Definir arquivo de Log de Vales", "log_vales_lpg.xlsx")
        if filepath:
            self.log_filepath = filepath
            display_path = os.path.basename(filepath)
            if len(self.log_filepath) > 40:
                display_path = f"...{self.log_filepath[-40:]}"
            self.log_file_label.config(text=display_path, fg="black")

    def log_single_valley_data(self):
        """Adiciona o vale ÚNICO atual como uma nova linha no arquivo de log."""
        if self.active_valley_wl is None:
            messagebox.showwarning("Sem Dados", "Nenhum vale detectado.")
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
        
        if self._append_to_log(df_new_row):
            messagebox.showinfo("Sucesso", f"Vale único registrado com sucesso em:\n{os.path.basename(self.log_filepath)}")
            
    def batch_process_and_log(self):
        """(v13) Processa TODOS os arquivos na lista e os registra no log."""
        
        # 1. Validações
        if not self.loaded_data:
            messagebox.showwarning("Sem Dados", "Nenhum arquivo carregado para processar em lote.")
            return

        params = self._get_filter_params()
        if params is None: return # Erro na validação
        window_size, poly_order, range_start, range_end, normalize = params

        if not self.log_filepath:
            messagebox.showwarning("Sem Log", "Defina um arquivo de Log primeiro.")
            self.set_log_file()
            if not self.log_filepath: return
        
        base_sample_name = self.sample_name_entry.get()
        if not base_sample_name:
            messagebox.showwarning("Sem Amostra", "Por favor, insira um 'Nome da Amostra' base para o lote.")
            return
            
        if not messagebox.askyesno("Confirmar Lote", f"Você está prestes a processar e registrar {len(self.loaded_data)} arquivos no log.\n\nArquivo de Log: {os.path.basename(self.log_filepath)}\nAmostra Base: {base_sample_name}\n\nContinuar?"):
            return

        # 2. Configura UI para processamento
        self.progress_bar['value'] = 0
        self.progress_bar['maximum'] = len(self.loaded_data)
        self.progress_label.config(text=f"Iniciando lote de {len(self.loaded_data)} arquivos...")
        self.master.update_idletasks()
        
        batch_results_list = []
        time_series_plot_data = [] # (index, valley_wl, valley_intensity)
        
        # 3. Loop de Processamento
        try:
            filenames = list(self.loaded_data.keys()) # Pega a ordem da lista
            for i, filename in enumerate(filenames):
                data = self.loaded_data[filename]
                wavelengths = data['wavelength']
                intensities = data['intensity']
                
                # Aplica filtro
                data_to_filter = intensities.copy()
                if normalize:
                    data_to_filter = data_to_filter - np.max(data_to_filter)
                
                sinal_filtrado = savgol_filter(data_to_filter, window_size, poly_order)
                
                # Encontra vale
                valley_result = self._find_valley(wavelengths, sinal_filtrado, range_start, range_end)
                
                if valley_result:
                    valley_wl, valley_intensity = valley_result
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] # Timestamp com milissegundos
                    
                    new_row = {
                        'horario': timestamp,
                        'comprimento_onda_filtrado (nm)': valley_wl,
                        'intensidade_filtrada_vale (dB)': valley_intensity,
                        'amostra': base_sample_name, # Nome da amostra é o mesmo para todo o lote
                        'arquivo_origem': filename
                    }
                    batch_results_list.append(new_row)
                    time_series_plot_data.append( (i, valley_wl, valley_intensity) )
                
                # Atualiza UI
                self.progress_bar['value'] = i + 1
                self.progress_label.config(text=f"Processando: {filename}")
                self.master.update_idletasks()

        except Exception as e:
            messagebox.showerror("Erro no Processamento em Lote", f"Ocorreu um erro durante o processamento:\n{e}")
            self.progress_label.config(text="Erro no lote.")
            return

        # 4. Salva resultados
        if not batch_results_list:
            messagebox.showwarning("Nenhum Resultado", "Processamento concluído, mas nenhum vale foi encontrado na faixa especificada.")
            self.progress_label.config(text="Lote concluído. Nenhum vale encontrado.")
            return
            
        df_batch = pd.DataFrame(batch_results_list)
        
        if self._append_to_log(df_batch):
            self.progress_label.config(text=f"Lote concluído! {len(df_batch)} vales registrados.")
            messagebox.showinfo("Lote Concluído", f"{len(df_batch)} vales foram processados e registrados com sucesso em:\n{os.path.basename(self.log_filepath)}")
            
            # 5. Plota a análise temporal
            self.last_batch_results = time_series_plot_data
            self._plot_time_series(self.last_batch_results)
            
        else:
            self.progress_label.config(text="Erro ao salvar o log do lote.")
            
    def save_plot_image(self):
        try:
            # Verifica qual separador está ativo
            if self.notebook.index(self.notebook.select()) == 0:
                # Separador Espectro
                self.canvas.toolbar.save_figure()
            else:
                # Separador Análise Temporal
                self.ts_canvas.toolbar.save_figure()
        except Exception as e:
            messagebox.showerror("Erro ao Salvar Imagem", f"Ocorreu um erro: {e}")

    def save_filtered_plot_only(self):
        if self.active_filtered_intensity is None:
            messagebox.showwarning("Sem Dados", "Nenhum dado filtrado para salvar.")
            return

        suggested_filename = f"{os.path.splitext(self.active_filename)[0]}_grafico_filtrado.png"
        filepath = filedialog.asksaveasfilename(
            title="Salvar Imagem Apenas do Filtro",
            initialfile=suggested_filename,
            # ADICIONA defaultextension para forçar a extensão
            defaultextension=".png",
            filetypes=[("Imagem PNG", "*.png"), ("Imagem PDF", "*.pdf"), ("Imagem SVG", "*.svg"), ("Todos os arquivos", "*.*")]
        )
        if not filepath: return

        try:
            fig_temp, ax_temp = plt.subplots(figsize=(10, 6))
            ax_temp.plot(self.active_wavelength, self.active_filtered_intensity, '-', 
                         color=self.color_filtrado, linewidth=2, label="Sinal Filtrado")

            if self.include_annotation_var.get() and self.active_valley_wl is not None:
                wv_min = self.active_valley_wl; int_min = self.active_valley_intensity
                text_label = f"Vale: {int_min:.2f} dB\n@ {wv_min:.2f} nm"
                xlim_temp = ax_temp.get_xlim()
                ax_temp.annotate(text_label, xy=(wv_min, int_min),
                    xytext=(wv_min + (xlim_temp[1] - xlim_temp[0]) * 0.05, int_min + abs(int_min)*0.1),
                    ha='left', va='bottom',
                    arrowprops=dict(arrowstyle='->', color=self.color_filtrado, connectionstyle='arc3,rad=0.3'),
                    bbox=dict(boxstyle='round,pad=0.3', fc=self.color_filtrado, alpha=0.2)
                )

            if self.include_range_var.get():
                try:
                    range_start = float(self.range_start_entry.get()) if self.range_start_entry.get() else None
                    range_end = float(self.range_end_entry.get()) if self.range_end_entry.get() else None
                    plot_ymin, plot_ymax = ax_temp.get_ylim()
                    if range_start: ax_temp.vlines(range_start, plot_ymin, plot_ymax, colors='blue', linestyles='dashed', alpha=0.5)
                    if range_end: ax_temp.vlines(range_end, plot_ymin, plot_ymax, colors='blue', linestyles='dashed', alpha=0.5)
                except Exception: pass

            ax_temp.set_title(f"Espectro Filtrado de: {self.active_filename}")
            ax_temp.set_xlabel("Comprimento de Onda (nm)")
            ax_temp.set_ylabel("Potência (dB)")
            ax_temp.legend(); ax_temp.grid(True, linestyle=':', alpha=0.7)
            
            main_xlim = self.ax.get_xlim(); main_ylim = self.ax.get_ylim()
            ax_temp.set_xlim(main_xlim); ax_temp.set_ylim(main_ylim)

            fig_temp.tight_layout()
            fig_temp.savefig(filepath, dpi=300)
            plt.close(fig_temp)
            messagebox.showinfo("Sucesso", f"Imagem (apenas filtro) salva em:\n{filepath}")

        except Exception as e:
            messagebox.showerror("Erro ao Salvar Imagem", f"Não foi possível salvar a imagem:\n{e}")
            if 'fig_temp' in locals(): plt.close(fig_temp)


if __name__ == "__main__":
    root = tk.Tk()
    app = LpgFilterApp(root)
    root.mainloop()