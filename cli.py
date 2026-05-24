import argparse
import json
import os
import sys
from pathlib import Path
from src.indexacion import IndiceInvertido
from src.modelos_clasicos import MotorClasico
from src.dataloader import cargar_corpus_completo

# Configuración
CONFIG = {
    'RUTA_INDICE': "db/indice_invertido.pkl",
    'RUTA_DATA': "data",
    'MODELOS_DISPONIBLES': ['tfidf', 'booleano', 'bm25']
}

# Emojis reutilizables
ICONS = {
    'init': '📚',
    'search': '🔍',
    'info': '📋',
    'models': '📊',
    'interactive': '🎯',
    'eval': '📊',
    'doc': '📄',
    'ok': '✓',
    'error': '❌',
    'warning': '⚠️',
    'bye': '👋',
}


class RICLI:
    """CLI para el Sistema de Recuperación de Información"""
    
    def __init__(self):
        self.motor = None
        self.corpus = None
        self.indice = None
        self.config = CONFIG
    
    def _print_status(self, status, message):
        """Imprime un mensaje con estado"""
        emoji = ICONS.get(status, '')
        if status == 'error':
            emoji = ICONS['error']
            print(f"{emoji} {message}", file=sys.stderr)
        else:
            emoji = ICONS.get(status, '')
            print(f"{emoji} {message}")
    
    def inicializar(self, verbose=True):
        """Carga o construye el índice e inicializa el motor"""
        if verbose:
            self._print_status('init', "Inicializando Sistema de Recuperación de Información...")
        
        try:
            idx_path = self.config['RUTA_INDICE']
            data_path = self.config['RUTA_DATA']
            
            # Cargar o construir índice
            if os.path.exists(idx_path) and os.path.getsize(idx_path) > 0:
                if verbose:
                    self._print_status('ok', f"Cargando índice desde: {idx_path}")
                self.indice = IndiceInvertido.cargar_de_disco(idx_path)
            else:
                if verbose:
                    self._print_status('ok', f"Construyendo índice desde: {data_path}")
                self.corpus, _ = cargar_corpus_completo(data_path)
                self.indice = IndiceInvertido()
                self.indice.construir_indice(self.corpus)
                self.indice.guardar_en_disco(idx_path)
                if verbose:
                    self._print_status('ok', f"Índice guardado en: {idx_path}")
            
            # Cargar corpus si no está cargado
            if not self.corpus:
                self.corpus, _ = cargar_corpus_completo(data_path)
            
            # Inicializar motor
            self.motor = MotorClasico(self.indice)
            if verbose:
                self._print_status('ok', "Motor de búsqueda listo\n")
            
            return True
        
        except Exception as e:
            self._print_status('error', f"Inicialización fallida: {e}")
            return False
    
    def _ejecutar_busqueda(self, consulta, modelo):
        """Ejecuta búsqueda según el modelo"""
        metodos = {
            'tfidf': 'buscar_coseno_tfidf',
            'booleano': 'buscar_booleano',
            'bm25': 'buscar_bm25',
        }
        
        if modelo not in metodos:
            raise ValueError(f"Modelo no disponible: {modelo}")
        
        try:
            metodo = getattr(self.motor, metodos[modelo])
            return metodo(consulta, top_k=5)
        except AttributeError:
            raise ValueError(f"Método no implementado en MotorClasico: {metodos[modelo]}")
    
    def buscar(self, consulta, top_k=5, modelo='tfidf'):
        """Realiza una búsqueda en el corpus"""
        if not self.motor:
            self._print_status('error', "Motor no inicializado. Ejecuta 'init' primero.")
            return
        
        print(f"\n{ICONS['search']} Buscando: '{consulta}'")
        print(f"{ICONS['models']} Modelo: {modelo.upper()} | Top-K: {top_k}\n")
        
        try:
            resultados = self._ejecutar_busqueda(consulta, modelo)
            
            if not resultados:
                self._print_status('warning', "No se encontraron resultados.")
                return
            
            # Mostrar resultados
            self._mostrar_resultados(resultados)
            
        except (AttributeError, ValueError) as e:
            self._print_status('error', f"Búsqueda fallida: {e}")
        except Exception as e:
            self._print_status('error', f"Error: {e}")
    
    def _mostrar_resultados(self, resultados):
        """Formatea y muestra los resultados de búsqueda"""
        print(f"{'Puesto':<6} {'Doc ID':<10} {'Score':<10} {'Preview':<50}")
        print("-" * 80)
        
        for puesto, (doc_id, score) in enumerate(resultados, 1):
            preview = self._obtener_preview(doc_id)
            print(f"{puesto:<6} {str(doc_id):<10} {score:<10.4f} {preview}")
        
        print()
    
    def _obtener_preview(self, doc_id, max_len=47):
        """Obtiene un preview del documento"""
        if not self.corpus or doc_id not in self.corpus:
            return ""
        
        text = str(self.corpus[doc_id])[:max_len]
        return text.replace('\n', ' ')
    
    def _print_separador(self, char="=", size=60):
        """Imprime un separador"""
        print(char * size)
    
    def info(self):
        """Muestra información del sistema"""
        print("\n" + "="*60)
        print(f"{ICONS['info']} INFORMACIÓN DEL SISTEMA")
        self._print_separador()
        
        # Índice
        if self.indice:
            print(f"{ICONS['ok']} Índice: {self.config['RUTA_INDICE']}")
            print(f"{ICONS['ok']} Términos únicos: {len(self.indice.indice)}")
        else:
            print(f"{ICONS['error']} Índice: No cargado")
        
        # Corpus
        if self.corpus:
            print(f"{ICONS['ok']} Corpus: {len(self.corpus)} documentos")
        else:
            print(f"{ICONS['error']} Corpus: No cargado")
        
        # Directorios
        print(f"\n{ICONS['info']} Directorios:")
        print(f"   - Datos: {self.config['RUTA_DATA']}")
        print(f"   - Base de datos: db/")
        self._print_separador()
        print()
    
    def listar_modelos(self):
        """Lista los modelos disponibles"""
        print(f"\n{ICONS['models']} MODELOS DISPONIBLES:\n")
        
        modelos_info = {
            'tfidf': 'TF-IDF con similaridad coseno',
            'booleano': 'Búsqueda booleana (AND, OR, NOT)',
            'bm25': 'BM25 (Okapi) - Ranking probabilístico',
        }
        
        for modelo, desc in modelos_info.items():
            print(f"  • {modelo:<12} - {desc}")
        
        print()
    
    def modo_interactivo(self):
        """REPL interactivo para búsquedas"""
        if not self.inicializar(verbose=False):
            return
        
        print(f"\n{ICONS['interactive']} MODO INTERACTIVO - Sistema de Recuperación")
        self._print_separador()
        print("Escribe 'ayuda' para ver comandos o 'salir' para terminar\n")
        
        config_sesion = {'modelo': 'tfidf', 'top_k': 5}
        comandos = self._crear_manejadores_comandos(config_sesion)
        
        while True:
            try:
                entrada = input("📌 > ").strip().lower()
                
                if not entrada:
                    continue
                
                # Procesar comando
                cmd = entrada.split()[0] if entrada.split() else None
                args = entrada.split()[1:] if len(entrada.split()) > 1 else []
                
                if cmd in comandos:
                    comandos[cmd](args, config_sesion)
                elif cmd == 'salir':
                    print(f"{ICONS['bye']} ¡Hasta luego!")
                    break
                else:
                    # Asumir búsqueda
                    self.buscar(entrada, top_k=config_sesion['top_k'], modelo=config_sesion['modelo'])
            
            except KeyboardInterrupt:
                print(f"\n{ICONS['bye']} Cancelado.")
                break
            except Exception as e:
                self._print_status('error', str(e))
    
    def _crear_manejadores_comandos(self, config):
        """Crea los manejadores de comandos del REPL"""
        return {
            'ayuda': lambda args, cfg: self._cmd_ayuda(),
            'modelo': lambda args, cfg: self._cmd_modelo(args, cfg),
            'top': lambda args, cfg: self._cmd_top(args, cfg),
            'info': lambda args, cfg: self.info(),
            'modelos': lambda args, cfg: self.listar_modelos(),
        }
    
    def _cmd_ayuda(self):
        """Mostrar ayuda"""
        print("\n" + "-"*60)
        print("📚 COMANDOS DISPONIBLES:")
        print("-"*60)
        print("  <consulta>      - Realiza una búsqueda")
        print("  modelo <nombre> - Cambia modelo (tfidf, booleano, bm25)")
        print("  top <número>    - Establece número de resultados")
        print("  info            - Muestra información del sistema")
        print("  modelos         - Lista modelos disponibles")
        print("  ayuda           - Muestra esta ayuda")
        print("  salir           - Termina el programa")
        print("-"*60 + "\n")
    
    def _cmd_modelo(self, args, config):
        """Cambiar modelo"""
        if args and args[0] in self.config['MODELOS_DISPONIBLES']:
            config['modelo'] = args[0]
            print(f"{ICONS['ok']} Modelo: {args[0]}\n")
        else:
            self._print_status('error', f"Modelo inválido. Disponibles: {', '.join(self.config['MODELOS_DISPONIBLES'])}")
    
    def _cmd_top(self, args, config):
        """Cambiar top-k"""
        try:
            if args and args[0].isdigit():
                config['top_k'] = int(args[0])
                print(f"{ICONS['ok']} Top-K: {args[0]}\n")
            else:
                raise ValueError("Debe ser un número")
        except ValueError as e:
            self._print_status('error', f"Top-K inválido: {e}")
    
    def evaluar_sistema(self, archivo_qrels='data/qrels.json'):
        """Evalúa el sistema usando QREL"""
        if not self.motor and not self.inicializar():
            return
        
        if not os.path.exists(archivo_qrels):
            self._print_status('error', f"Archivo no encontrado: {archivo_qrels}")
            return
        
        try:
            with open(archivo_qrels, 'r') as f:
                qrels = json.load(f)
            
            print(f"\n{ICONS['eval']} Evaluando {len(qrels)} queries\n")
            
            metricas = self._calcular_metricas(qrels)
            self._mostrar_metricas(metricas)
            
        except json.JSONDecodeError:
            self._print_status('error', f"JSON inválido: {archivo_qrels}")
        except Exception as e:
            self._print_status('error', f"Error: {e}")
    
    def _calcular_metricas(self, qrels):
        """Calcula métricas Precision, Recall, F1"""
        metricas = []
        
        for query_id, relevant_docs in qrels.items():
            resultados = self.motor.buscar_coseno_tfidf(query_id, top_k=5)
            retrieved = [doc_id for doc_id, _ in resultados]
            
            if relevant_docs:
                tp = len(set(retrieved) & set(relevant_docs))
                precision = tp / len(retrieved) if retrieved else 0
                recall = tp / len(relevant_docs) if relevant_docs else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                metricas.append({
                    'query': query_id[:15],
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
        
        return metricas
    
    def _mostrar_metricas(self, metricas):
        """Muestra tabla de métricas"""
        if not metricas:
            self._print_status('warning', "Sin resultados")
            return
        
        print(f"{'Query':<20} {'Prec@5':<10} {'Rec@5':<10} {'F1@5':<10}")
        self._print_separador("-", 50)
        
        for m in metricas:
            print(f"{m['query']:<20} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1']:<10.4f}")
        
        # Calcular promedios
        self._print_separador("-", 50)
        n = len(metricas)
        avg_p = sum(m['precision'] for m in metricas) / n
        avg_r = sum(m['recall'] for m in metricas) / n
        avg_f1 = sum(m['f1'] for m in metricas) / n
        print(f"{'PROMEDIO':<20} {avg_p:<10.4f} {avg_r:<10.4f} {avg_f1:<10.4f}\n")
    
    def mostrar_documento(self, doc_id):
        """Muestra el contenido de un documento"""
        if not self.corpus:
            self._print_status('error', "Corpus no cargado")
            return
        
        if doc_id not in self.corpus:
            self._print_status('error', f"Documento no encontrado: {doc_id}")
            return
        
        print(f"\n{'='*70}")
        print(f"{ICONS['doc']} DOCUMENTO: {doc_id}")
        print('='*70 + "\n")
        print(str(self.corpus[doc_id]))
        print(f"\n{'='*70}\n")


def crear_parser():
    """Crea el parser de argumentos"""
    parser = argparse.ArgumentParser(
        prog='Sistema de Recuperación de Información',
        description="🔍 CLI para el Sistema IR Reuters-21578",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EJEMPLOS:
  python cli.py init
  python cli.py search "financial news" --top 10
  python cli.py search "world bank" --modelo tfidf
  python cli.py info
  python cli.py interactive
  python cli.py eval
  python cli.py doc 257
        """
    )
    
    subparsers = parser.add_subparsers(dest='comando')
    
    # init
    subparsers.add_parser('init', help='Inicializar índice')
    
    # search
    search = subparsers.add_parser('search', help='Buscar')
    search.add_argument('consulta', help='Texto a buscar')
    search.add_argument('--modelo', '-m', default='tfidf', 
                       choices=['tfidf', 'booleano', 'bm25'], help='Modelo (default: tfidf)')
    search.add_argument('--top', '-k', type=int, default=5, help='Resultados (default: 5)')
    
    # info
    subparsers.add_parser('info', help='Información del sistema')
    
    # modelos
    subparsers.add_parser('modelos', help='Listar modelos')
    
    # interactive
    subparsers.add_parser('interactive', help='Modo interactivo')
    
    # eval
    eval_cmd = subparsers.add_parser('eval', help='Evaluar sistema')
    eval_cmd.add_argument('--qrels', default='data/qrels.json', help='Ruta a qrels')
    
    # doc
    doc = subparsers.add_parser('doc', help='Ver documento')
    doc.add_argument('doc_id', help='ID del documento')
    
    return parser


def main():
    """Función principal"""
    parser = crear_parser()
    args = parser.parse_args()
    
    if not args.comando:
        parser.print_help()
        return
    
    cli = RICLI()
    
    # Comandos que necesitan inicialización
    cmd_requiere_init = {'search', 'info', 'doc'}
    
    try:
        if args.comando == 'init':
            cli.inicializar()
        
        elif args.comando in cmd_requiere_init:
            if not cli.inicializar():
                return
            
            if args.comando == 'search':
                cli.buscar(args.consulta, top_k=args.top, modelo=args.modelo)
            elif args.comando == 'info':
                cli.info()
            elif args.comando == 'doc':
                cli.mostrar_documento(args.doc_id)
        
        elif args.comando == 'modelos':
            cli.listar_modelos()
        
        elif args.comando == 'interactive':
            cli.modo_interactivo()
        
        elif args.comando == 'eval':
            cli.evaluar_sistema(args.qrels)
    
    except KeyboardInterrupt:
        print(f"\n{ICONS['bye']} Cancelado")
    except Exception as e:
        print(f"{ICONS['error']} Error: {e}", file=sys.stderr)


if __name__ == '__main__':
    main()
