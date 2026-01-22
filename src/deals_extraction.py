from dealabs import Dealabs
import json
import time 
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- FONCTION WORKER MODIFIÉE ---
def fetch_deal_comments(deal):
    """
    Récupère les commentaires.
    CRUCIAL : On instancie le client ICI pour qu'il soit unique à ce thread.
    """
    # Chaque thread crée sa propre connexion.
    # Cela évite que les threads se bloquent entre eux sur une session partagée.
    local_client = Dealabs() 
    
    thread_id = deal.get('thread_id') or deal.get('id')
    try:
        comments = local_client.get_thread_comments(thread_id)
        deal['comments'] = [c.to_dict() if hasattr(c, 'to_dict') else c for c in comments]
    except Exception as err:
        deal['comments'] = []
    
    return str(thread_id), deal

def main():
    # On garde une instance principale juste pour le listing des pages
    main_client = Dealabs()
    
    # --- CONFIGURATION ---
    nb_deals_to_fetch = 100000
    batch_size_pages = 50
    per_page = 50           
    
    # --- VITESSE ---
    # Puisque chaque worker a son client, vous pouvez monter plus haut.
    # Essayez 20. Si erreurs 429/Ban, redescendez à 10.
    MAX_WORKERS = 30  
    # ---------------------

    batch_deals = []          
    total_processed = 0       
    page = 0                  
    pages_in_current_batch = 0 
    file_part = 1
    batch_id = 1
    
    global_start_time = time.time()
    batch_start_time = time.time()
    
    print(f"Démarrage avec {MAX_WORKERS} workers INDÉPENDANTS.")

    while total_processed < nb_deals_to_fetch:
        try:
            # 1. METADATA (Séquentiel avec le client principal)
            #results = main_client.search_deals(params = {'group_id': 2, 'order_by': 'new'})
            response = main_client.get_new_deals(params={'page': page, 'limit': per_page})
            deals_data = response.get('data', [])
            
            if not deals_data:
                print("Plus de deals disponibles.")
                if batch_deals: pass 
                else: break
                
            batch_deals.extend(deals_data)
            page += 1
            pages_in_current_batch += 1 
            
            current_total = total_processed + len(batch_deals)
            
            # 2. DÉCLENCHEMENT
            if pages_in_current_batch >= batch_size_pages or current_total >= nb_deals_to_fetch or not deals_data:
                
                listing_duration = time.time() - batch_start_time
                
                if current_total > nb_deals_to_fetch:
                    to_keep = nb_deals_to_fetch - total_processed
                    batch_deals = batch_deals[:to_keep]

                print(f"\n--- BATCH COMPLET n°{batch_id} ({len(batch_deals)} deals) ---")
                print(">>> Lancement du téléchargement parallèle...")

                processing_start_time = time.time()

                deals_by_thread_id = {}
                
                # 3. RÉCUPÉRATION PARALLÈLE
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    # NOTE : On ne passe plus 'client' en argument, le worker crée le sien
                    future_to_deal = {executor.submit(fetch_deal_comments, deal): deal for deal in batch_deals}
                    
                    completed_count = 0
                    for future in as_completed(future_to_deal):
                        t_id, processed_deal = future.result()
                        deals_by_thread_id[t_id] = processed_deal
                        
                        completed_count += 1
                        if completed_count % 50 == 0:
                            elapsed = time.time() - processing_start_time
                            rate = completed_count / elapsed if elapsed > 0 else 0
                            print(f"   Progression: {completed_count}/{len(batch_deals)} ({rate:.1f} deals/sec)")

                # 4. ÉCRITURE
                filename = f"deals_dump_part_{file_part}.json"
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(deals_by_thread_id, f, ensure_ascii=False, indent=2)
                
                processing_duration = time.time() - processing_start_time
                total_batch_duration = listing_duration + processing_duration
                
                print(f">>> Fichier {filename} généré.")
                print(f"--- STATS BATCH n°{batch_id} ---")
                print(f"Listing pages  : {str(timedelta(seconds=int(listing_duration)))}")
                print(f"Comms (Multi)  : {str(timedelta(seconds=int(processing_duration)))}")
                print(f"Vitesse moy.   : {len(batch_deals)/processing_duration:.2f} deals/sec")
                print("-----------------------------")
                
                # 5. RESET
                total_processed += len(batch_deals) 
                batch_deals = []                    
                pages_in_current_batch = 0          
                file_part += 1
                batch_id += 1     
                batch_start_time = time.time()
                
                if total_processed >= nb_deals_to_fetch:
                    break 
                
                print(f"--- Reprise du listing ---")
                time.sleep(1) 

        except Exception as e:
            print(f"Erreur CRITIQUE : {e}")
            break

    total_duration = time.time() - global_start_time
    print(f"Opération terminée. Temps total : {str(timedelta(seconds=int(total_duration)))}")

if __name__ == "__main__":
    main()