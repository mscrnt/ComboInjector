# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 10:12:02 2024

ComboInjector class for managing combos, actions, and special moves
in a fighting game environment. Supports configurable probabilities for
different action categories (jump/movement/basic/combos), error handling, and
improved readability

@author: ruhe
"""

from collections import deque
import numpy as np

class ComboInjector(object):
    """
    A utility class for injecting combos, special moves, and basic actions
    for a fighting game environment. Currently supports 'sfiii' with multi-discrete
    action spaces.

    Attributes
    ----------
    environment_name : str
        Name of the environment (default 'sfiii').
    mode : str
        Action mode (must be 'multi_discrete' currently).
    characters : dict
        Dict of game_name -> list of character names.
    base_movement_names : dict
        Mapping of short keys ('l', 'r', etc.) to movement arrays.
    base_attack_names : dict
        Mapping of short keys ('lp', 'hp', etc.) to attack arrays.
    action_idx_lookup : dict
        Maps string combos like 'l+lp' to integer action indices.
    input_lookup : dict
        Maps an integer action index back to the final multi-discrete array.
    move_pattern_names : dict
        Named movement patterns for quarter circles, half circles, etc.
    move_sequence : dict
        Tracks the queued moves for each player.

    Methods
    -------
    reset([character_1, ...], [super_art_1, ...]):
        Clear stored move sequences for each agent/player and initialize agent states.
    in_sequence(player: str) -> bool:
        Returns True if the specified player still has queued moves.
    action_space_size() -> int:
        Returns the size of the known action space (if relevant).
    sample_character_special(character: str, super_art: int) -> List[str]:
        Samples a special move sequence for a given character & super art.
    sample_movement_action() -> List[str]:
        Samples a basic movement pattern or repeated direction inputs.
    sample(combo: bool = True, 
           prob_jump: float = 0.04, 
           prob_basic: float = 0.25, 
           prob_combos: float = 0.70, 
           prob_movement: float = 0.40,
           vs: bool = False) -> Union[int, tuple]:
        Main entry point to sample an action (or tuple of actions) for single / vs mode.
    """
    
    def __init__(self,
                 environment_name: str = 'sfiii',
                 mode: str = 'multi_discrete',
                 frame_skip: int = 4):
        """
        Initialize the ComboInjector.

        Parameters
        ----------
        environment_name : str, optional
            Name of the environment (default is 'sfiii').
        mode : str, optional
            Action mode (must be 'multi_discrete'); future expansion possible.
        """

        self.environment_name = environment_name
        self.mode = mode
        self.frame_skip = frame_skip

        # Only support multi_discrete for now
        if self.mode != 'multi_discrete':
            raise ValueError("Only 'multi_discrete' mode is currently supported.")
            
        # Characters available for the environment
        self.characters = {
            'sfiii': {
                "Alex" : {'power_bomb' : {'prob': 0.15, 'combo_str' : 'comb_hc_p'},
                          'spiral_ddt' : {'prob' : 0.15, 'combo_str' : 'comb_hc_k'},
                          'flash_chop' : {'prob' : 0.15, 'combo_str' : 'comb_qc_p'},
                          'air_knee_smash' : {'prob' : 0.15, 'combo_str' : 'comb_dp_k'},
                          'air_stampede' : {'prob' : 0.15, 'combo_str' : 'hold_d_16_64_k'},
                          'slash_elbow' : {'prob' : 0.15, 'combo_str' : 'hold_lr_16_64_k'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_fc_mp',
                                         'combo_str_2' : 'comb_2qc_mp',
                                         'combo_str_3' : 'comb_2qc_mp',}
                          },
                "Twelve" : {'ndl' : {'prob': 0.3, 'combo_str' : 'comb_qc_p'},
                            'axe' : {'prob': 0.3, 'combo_str' : 'comb_qc_p/rep_p_3_12_t'},
                            'dra' : {'prob' : 0.3, 'combo_str' : 'comb_qc_k'},
                            'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp',
                                         'combo_str_2' : 'comb_2qc_mk',
                                         'combo_str_3' : 'comb_2qc_mp',}
                          },
                "Hugo" : {'shootdown_backbreaker' : {'prob': 0.15, 'combo_str' : 'comb_dp_k'},
                            'ultra_throw' : {'prob': 0.15, 'combo_str' : 'comb_hc_k'},
                            'moonsault_press' : {'prob' : 0.15, 'combo_str' : 'comb_fc_p'},
                            'meat_squasher' : {'prob' : 0.15, 'combo_str' : 'comb_fc_k'},
                            'giant_palm_bomber' : {'prob' : 0.15, 'combo_str' : 'comb_qc_p'},
                            'monster_lariat' : {'prob' : 0.15, 'combo_str' : 'comb_qc_k'},
                            'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2fc_mp',
                                         'combo_str_2' : 'comb_2qc_mk',
                                         'combo_str_3' : 'comb_2qc_mp/rep_mp_0_8_',}
                          },
                "Sean" : {'zenten' : {'prob': 0.18, 'combo_str' : 'comb_qc_p'},
                          'sean_tackle' : {'prob' : 0.18, 'combo_str' : 'comb_hc_p/rep_p_0_8_'},
                          'dragon_smash' : {'prob' : 0.18, 'combo_str' : 'comb_dp_p'},
                          'tornado_ryuubi_kyaku' : {'prob' : 0.36, 'combo_str' : 'comb_qc_k'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp',
                                         'combo_str_2' : 'comb_2qc_mp/rep_mp_0_12_t',
                                         'combo_str_3' : 'comb_2qc_mp',}
                          },
                "Makoto" : {'karakusa' : {'prob': 0.18, 'combo_str' : 'comb_hc_k'},
                          'hayate_oroshi' : {'prob' : 0.36, 'combo_str' : 'comb_qc_p/rep_p_0_8_'},
                          'fukiage' : {'prob' : 0.18, 'combo_str' : 'comb_dp_p'},
                          'tsurugi' : {'prob' : 0.18, 'combo_str' : 'comb_qc_k'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp',
                                         'combo_str_2' : 'comb_2qc_mk',
                                         'combo_str_3' : 'comb_2qc_mp',}
                          },
                "Elena" : {'rhino_horn' : {'prob': 0.18, 'combo_str' : 'comb_hc_k'},
                          'mallet_smash' : {'prob' : 0.18, 'combo_str' : 'comb_hc_p'},
                          'spin_scythe' : {'prob' : 0.18, 'combo_str' : 'comb_qc_k'},
                          'scratch_wheel_lynx_tail' : {'prob' : 0.36, 'combo_str' : 'comb_dp_k'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mk',
                                         'combo_str_2' : 'comb_2qc_mk',
                                         'combo_str_3' : 'comb_2qc_mp',}
                          },
                "Ibuki" : {'raida' : {'prob': 0.18, 'combo_str' : 'comb_hc_p'},
                          'kasumi_gake_tsumuji' : {'prob' : 0.18, 'combo_str' : 'comb_qc_k'},
                          'tsuji_goe' : {'prob' : 0.18, 'combo_str' : 'comb_dp_p'},
                          'kunai_kubi_ori' : {'prob' : 0.18, 'combo_str' : 'comb_qc_p'},
                          'kazekiri_hien' : {'prob' : 0.18, 'combo_str' : 'comb_dp_k'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp/rep_mp_0_16_t',
                                         'combo_str_2' : 'comb_2qc_mp',
                                         'combo_str_3' : 'comb_2qc_mp',}
                          },
                "Chun-Li" : {'kikoken' : {'prob': 0.18, 'combo_str' : 'comb_hc_p'},
                          'hazanshu' : {'prob' : 0.18, 'combo_str' : 'comb_hc_k'},
                          'spinning_bird_kick' : {'prob' : 0.18, 'combo_str' : 'hold_d_16_64_k'},
                          'hyakuretsu_kyaku' : {'prob' : 0.18, 'combo_str' : 'rep_k_3_16_t'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp',
                                         'combo_str_2' : 'comb_2qc_mk',
                                         'combo_str_3' : 'comb_2qc_mk',}
                          },
                "Dudley" : {'ducking_straight_short_swing_blow' : {'prob': 0.3, 'combo_str' : 'comb_hc_k/rep_p_0_4_t'},
                            'ducking_upper_short_swing_blow' : {'prob': 0.3, 'combo_str' : 'comb_hc_k/rep_k_0_4_t'},
                          'machine_gun_blow_cross_counter' : {'prob' : 0.3, 'combo_str' : 'comb_hc_p'},
                          'jet_upper' : {'prob' : 0.18, 'combo_str' : 'comb_dp_p'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp',
                                         'combo_str_2' : 'comb_2qc_mp/rep_mp_3_12_t',
                                         'combo_str_3' : 'comb_2qc_mp',}
                          },
                "Necro" : {'snake_fang_rising_cobra' : {'prob': 0.18, 'combo_str' : 'comb_hc_k'},
                          'denji_blast' : {'prob' : 0.18, 'combo_str' : 'comb_dp_p/rep_p_0_12_t'},
                          'flying_viper' : {'prob' : 0.18, 'combo_str' : 'comb_qc_p'},
                          'rising_kobra' : {'prob' : 0.18, 'combo_str' : 'comb_qc_k'},
                          'tornado_hook' : {'prob' : 0.18, 'combo_str' : 'comb_hc_p'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp/rep_mp_0_16_t',
                                         'combo_str_2' : 'comb_2qc_mp',
                                         'combo_str_3' : 'comb_2qc_mp',}
                          },
                "Q" : {'capture_deadly_blow' : {'prob': 0.225, 'combo_str' : 'comb_hc_k'},
                          'dashing_head' : {'prob' : 0.225, 'combo_str' : 'hold_lr_16_64_p/rep_p_0_8_'},
                          'dashing_leg' : {'prob' : 0.225, 'combo_str' : 'hold_lr_16_64_k'},
                          'high_speed_barage' : {'prob' : 0.225, 'combo_str' : 'comb_qc_p'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp',
                                         'combo_str_2' : 'comb_2qc_mp',
                                         'combo_str_3' : 'comb_2qc_mp',}
                          },
                "Oro" : {'niu_riki' : {'prob': 0.225, 'combo_str' : 'comb_hc_p'},
                          'nichirin_shou' : {'prob' : 0.225, 'combo_str' : 'hold_lr_16_64_p'},
                          'oni_yanma' : {'prob' : 0.225, 'combo_str' : 'hold_d_16_64_p'},
                          'jinchuu_watari_hitobashira_nobori' : {'prob' : 0.225, 'combo_str' : 'comb_qc_k'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp/hold_lr_4_24_p',
                                         'combo_str_2' : 'comb_2qc_mp',
                                         'combo_str_3' : 'comb_2qc_mp',}
                          },
                "Urien" : {'metallic_sphere' : {'prob': 0.225, 'combo_str' : 'comb_qc_p/rep_p_0_12_'},
                          'chariot_tackle' : {'prob' : 0.225, 'combo_str' : 'hold_lr_16_64_k'},
                          'dangerous_headbutt' : {'prob' : 0.225, 'combo_str' : 'hold_d_16_64_p'},
                          'violence_knee_drop' : {'prob' : 0.225, 'combo_str' : 'hold_d_16_64_k'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp',
                                         'combo_str_2' : 'comb_2qc_mp',
                                         'combo_str_3' : 'comb_2qc_mp',}
                          },
                "Remy" : {'light_of_virtue' : {'prob': 0.225, 'combo_str' : 'hold_lr_16_64_p'},
                          'light_of_virtue_low' : {'prob' : 0.225, 'combo_str' : 'hold_lr_16_64_k'},
                          'rising_rage_flash' : {'prob' : 0.225, 'combo_str' : 'hold_d_16_128_k'},
                          'cold_blue_kick' : {'prob' : 0.225, 'combo_str' : 'comb_qc_k'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp',
                                         'combo_str_2' : 'comb_2qc_mk',
                                         'combo_str_3' : 'comb_2qc_mk',}
                          },
                "Ryu" : {'hadouken' : {'prob': 0.225, 'combo_str' : 'comb_qc_p'},
                          'shoryuken' : {'prob' : 0.225, 'combo_str' : 'comb_dp_p'},
                          'tatsumaki_senpukyaku' : {'prob' : 0.225, 'combo_str' : 'comb_qc_k'},
                          'joudan_sokutou_geri' : {'prob' : 0.225, 'combo_str' : 'comb_hc_k'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp',
                                         'combo_str_2' : 'comb_2qc_mp',
                                         'combo_str_3' : 'comb_2qc_mp/rep_mp_0_16_',}
                          },
                "Gouki" : {'go_zankuu_hadouken' : {'prob': 0.15, 'combo_str' : 'comb_qc_p'},
                          'shakenutsu-hadouken' : {'prob' : 0.15, 'combo_str' : 'comb_hc_p'},
                          'go_shoryuken' : {'prob' : 0.15, 'combo_str' : 'comb_dp_p'},
                          'tatsumaki_senpukyaku' : {'prob' : 0.15, 'combo_str' : 'comb_qc_k'},
                          'hyakkishu_go' : {'prob' : 0.075, 'combo_str' : 'comb_dp_k/rep_p_0_2_t'},
                          'hyakkishu_sho' : {'prob' : 0.075, 'combo_str' : 'comb_dp_k/rep_k_0_2_t'},
                          'hyakkishu_sho_sai' : {'prob' : 0.075, 'combo_str' : 'comb_dp_k/rep_mpk_0_2_t'},
                          'shun_goku_satsu' : {'prob' : 0.055, 'combo_str' : 'raw_+lp_+_+lp/comb_lr_/raw_+lk_+_+hp'},
                          'target_combo_1' : {'prob' : 0.02, 'combo_str' : 'rep_mp_5_5_t/raw_+/rep_hp_5_5_t'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp',
                                         'combo_str_2' : 'comb_2qc_mp',
                                         'combo_str_3' : 'comb_2qc_mk',}
                          },
                "Yun" : {'zenpou_tenshin' : {'prob': 0.225, 'combo_str' : 'comb_hc_k'},
                          'kobokushi_zesshou_hohou' : {'prob' : 0.225, 'combo_str' : 'comb_qc_p'},
                          'tetsuzanko' : {'prob' : 0.225, 'combo_str' : 'comb_dp_p'},
                          'nishoukyaku' : {'prob' : 0.225, 'combo_str' : 'comb_dp_k'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp',
                                         'combo_str_2' : 'comb_2qc_mp',
                                         'combo_str_3' : 'comb_2qc_mk',}
                          },
                "Yang" : {'tourou_zan_byakko_soushouda' : {'prob': 0.225, 'combo_str' : 'comb_qc_p'},
                          'senkyuutai' : {'prob' : 0.225, 'combo_str' : 'comb_qc_k'},
                          'zenpou_tenshin' : {'prob' : 0.225, 'combo_str' : 'comb_hc_k'},
                          'kaihou' : {'prob' : 0.225, 'combo_str' : 'comb_dp_k'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp',
                                         'combo_str_2' : 'comb_2qc_mk',
                                         'combo_str_3' : 'comb_2qc_mp',}
                          },
                "Ken" : {'hadouken' : {'prob': 0.225, 'combo_str' : 'comb_qc_p'},
                          'shoryuken' : {'prob' : 0.225, 'combo_str' : 'comb_dp_p'},
                          'tatsumaki_senpukyaku' : {'prob' : 0.225, 'combo_str' : 'comb_qc_k'},
                          'grab' : {'prob' : 0.225, 'combo_str' : 'rep_lpk_1_8_t'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp',
                                         'combo_str_2' : 'comb_2qc_mk/rep_mk_0_16_t',
                                         'combo_str_3' : 'comb_2qc_mk',}
                          },
            },
        }

        if self.mode != 'multi_discrete':
            raise NotImplementedError(
                f"Only 'multi_discrete' is supported, got mode={mode}"
            )

        # Base movement definitions (multi-discrete)
        # The first array entry is the directional input, second is always 0 for movement only.
        self.base_movement_names = {
            '':   np.array([0, 0]),
            'l':  np.array([1, 0]),
            'ul': np.array([2, 0]),
            'u':  np.array([3, 0]),
            'ur': np.array([4, 0]),
            'r':  np.array([5, 0]),
            'dr': np.array([6, 0]),
            'd':  np.array([7, 0]),
            'dl': np.array([8, 0]),
        }

        # Base attack definitions (multi-discrete)
        # The first array entry is always 0 for no directional component, second is the attack.
        self.base_attack_names = {
            '':    np.array([0, 0]),
            'lp':  np.array([0, 1]),
            'mp':  np.array([0, 2]),
            'hp':  np.array([0, 3]),
            'lk':  np.array([0, 4]),
            'mk':  np.array([0, 5]),
            'hk':  np.array([0, 6]),
            'lpk': np.array([0, 7]),
            'mpk': np.array([0, 8]),
            'hpk': np.array([0, 9]),
        }

        # Flatten the actions into a list of "movement+attack" combos for indexing
        self._base_actions = []
        for move_name, move_arr in self.base_movement_names.items():
            for attack_name, attack_arr in self.base_attack_names.items():
                combo_str = f'{move_name}+{attack_name}'
                self._base_actions.append(combo_str)
        # Map combo string -> index
        self.action_idx_lookup = {
            combo_str: i for i, combo_str in enumerate(self._base_actions)
        }

        # Reverse mapping: index -> multi-discrete array
        self.input_lookup = {}
        for combo_str, idx in self.action_idx_lookup.items():
            move_part, attack_part = combo_str.split('+')
            movement_arr = self.base_movement_names[move_part]
            attack_arr = self.base_attack_names[attack_part]
            self.input_lookup[idx] = (movement_arr + attack_arr).tolist()

        # Named movement patterns (quarter circles, half circles, etc.)
        self.move_pattern_names = {
            'rqc':  ['d', 'dr', 'r'],
            'lqc':  ['d', 'dl', 'l'],
            'rhc':  ['l', 'dl', 'd', 'dr', 'r'],
            'lhc':  ['r', 'dr', 'd', 'dl', 'l'],
            'rdp':  ['r', 'd', 'dr'],
            'ldp':  ['l', 'd', 'dl'],
            'rfc':  ['r', 'dr', 'd', 'dl', 'l', 'ul', 'u', 'ur'],
            'lfc':  ['l', 'dl', 'd', 'dr', 'r', 'ur', 'u', 'ul'],
            'sr':   ['r', '', 'r'],
            'sl':   ['l', '', 'l'],
            'sur':  ['d', 'ur'],
            'su':   ['d', 'u'],
            'sul':  ['d', 'ul'],
        }

        self.agent_state = {}
        
    def reset(self, characters, super_arts):
        """
        Clear stored move sequences for each agent (player).
        Typically called between episodes or after environment resets.
        """
        self.agent_state = {}
        for i, (character, super_art) in enumerate(zip(characters, super_arts)):
            if character not in self.characters[self.environment_name]:
                raise NotImplementedError(f"Character '{character}' not supported yet.")
            if super_art not in [1, 2, 3]:
                raise NotImplementedError(f"Super art '{super_art}' not supported.")
            self.agent_state[f'agent_{i}'] = {'move_sequence' : deque(),
                                               'character' : character,
                                               'super_art' : super_art}
        
    def in_sequence(self, player: str) -> bool:
        """
        Check if the specified player's sequence is non-empty.

        Parameters
        ----------
        player : str
            e.g. 'agent_0', 'agent_1', etc.

        Returns
        -------
        bool
            True if there are still queued moves, False otherwise.
        """
        return len(self.agent_state[player]['move_sequence']) > 0
    
    def _combine_actions(self, move_string, attack_string):
        if move_string == 'qc':
            m_seq = self.move_pattern_names[np.random.choice(['rqc', 'lqc'])]
        elif move_string == '2qc':
            m_seq = self.move_pattern_names[np.random.choice(['rqc', 'lqc'])] * 2
        elif move_string == 'hc':
            m_seq = self.move_pattern_names[np.random.choice(['rhc', 'lhc'])]
        elif move_string == 'fc':
            m_seq = self.move_pattern_names[np.random.choice(['rfc', 'lfc'])]
        elif move_string == '2fc':
            m_seq = self.move_pattern_names[np.random.choice(['rfc', 'lfc'])] * 2
        elif move_string == 'dp':
            m_seq = self.move_pattern_names[np.random.choice(['rdp', 'ldp'])]
        elif move_string == '2dp':
            m_seq = self.move_pattern_names[np.random.choice(['rdp', 'ldp'])] * 2
        elif move_string == 'lr':
            m_seq = [np.random.choice(['l', 'r'])]
        else:
            m_seq = [move_string]
        
        if attack_string == 'p':
            a = np.random.choice(['lp', 'mp', 'hp'])
        elif attack_string == 'k':
            a = np.random.choice(['lk', 'mk', 'hk'])
        else:
            a = attack_string

        a_seq = [''] * (len(m_seq) - 1) + [a]
        return [f"{i}+{ii}" for i, ii in zip(m_seq, a_seq)]
    
    def _hold_direction(self, direction, min_frame, max_frame, release=''):
        
        if release == 'p':
            release = np.random.choice(['lp', 'mp', 'hp'])
        elif release == 'k':
            release = np.random.choice(['lk', 'mk', 'hk'])
            
        num_steps = np.random.randint(int(min_frame)//self.frame_skip, (int(max_frame)+1)//self.frame_skip)
        
        if direction == 'd':
            m1 = np.random.choice(['d', 'dl', 'dr'])
            m2_seq = []
            if release:
                m2_seq = ['u']
            m_seq = [m1] * num_steps + m2_seq
            a_seq = [''] * (len(m_seq) - bool(release)) + [release] * bool(release)
            
        elif direction == 'lr':
            m1 = np.random.choice(['l', 'dl', 'dr', 'r'])
            m2_seq = []
            if release:
                m2_seq = ['l'] if m1.endswith('r') else ['r']
            m_seq = [m1] * num_steps + m2_seq
            a_seq = [''] * (len(m_seq) - bool(release)) + [release] * bool(release)

        return [f"{i}+{ii}" for i, ii in zip(m_seq, a_seq)]
    
    def _repeat_attack(self, attack_string, min_repeats, max_repeats, tap=''):
        tap = bool(tap)
        if attack_string == 'p':
            attack_string = np.random.choice(['lp', 'mp', 'hp'])
        elif attack_string == 'k':
            attack_string = np.random.choice(['lk', 'mk', 'hk'])
            
        seq = ['+' + attack_string] + ['+'] * tap
        return seq * np.random.randint(int(min_repeats), int(max_repeats)+1)
    
    def _raw_action_string(self, action_string):
        return action_string.split('_')
    
    def _decode_action_string(self, action_string):
        action_sequence = []
        for sub_string in action_string.split('/'):
            sub_string = sub_string.split('_')
            if sub_string[0] == 'comb':
                action_sequence += self._combine_actions(sub_string[1], sub_string[2])
            elif sub_string[0] == 'hold':
                action_sequence += self._hold_direction(*sub_string[1:])
            elif sub_string[0] == 'rep':
                action_sequence += self._repeat_attack(*sub_string[1:])
                
        return action_sequence
    
    def action_space_size(self) -> int:
        """
        Return the size of the known action space for multi-discrete combos.

        Returns
        -------
        int
            Number of possible "movement + attack" combos.
        """
        return len(self._base_actions)
        
    def sample_character_special(self, player) -> list:
        """
        Returns a list of combos in the form ['d+lp', 'dr+lp', 'r+lp', ...]
        for a particular character + super_art level.

        This method contains custom combos/roll logic for each character.

        Parameters
        ----------
        character : str
            Name of the character to sample from (e.g. 'Gouki', 'Chun-Li', etc.)
        super_art : int
            Indicates which super art is selected (1, 2, or 3), used in some combos.

        Returns
        -------
        list of str
            e.g. ['d+lp', 'dr+lp', 'r+lp'] or an empty list if no combos are found.

        Raises
        ------
        NotImplementedError
            If the character is not recognized or not implemented.
        """
        
        character = self.agent_state[player]['character']
        super_art = self.agent_state[player]['super_art']
        
        if character not in self.characters[self.environment_name]:
            raise NotImplementedError(f"Character '{character}' not supported yet.")
            
        roll = np.random.rand()
        prob_sum = 1
        for move, params in self.characters[self.environment_name][character].items():
            prob_sum -= params['prob']
            if roll >= prob_sum:
                suffix = ''
                if move == 'super_art':
                    suffix = f"_{super_art}"
                return self._decode_action_string(params['combo_str'+suffix])
        return [np.random.choice(self._base_actions)]
            
    def sample_movement_action(self, prob_dash=0.15, repeat_min=12, repeat_max=64) -> list:
        """
        Sample a basic movement action, possibly including quick repeated directions
        or short combos involving directions.

        Returns
        -------
        list of str
            e.g. ['r+', '+', 'r+'] or repeated directional combos.
        """
        roll = np.random.rand()
        # 15% chance to do some lateral movement, otherwise random repeated direction
        if roll < prob_dash:
            # e.g. 'r+' or 'l+', plus a filler '+'
            move = np.random.choice(['r+', 'l+'])
            return [move, '+', move]
        # Otherwise choose a repeated direction
        repeated = np.random.choice(['l+', 'dl+', 'd+', 'dr+', 'r+'])
        length = np.random.randint(repeat_min//self.frame_skip, repeat_max//self.frame_skip)
        return [repeated] * length
    
    def sample_jump_action(self, prob_super_jump=0.15) -> list:
        roll = np.random.rand()
        # 15% chance to do some lateral movement, otherwise random repeated direction
        if roll < prob_super_jump:
            # e.g. 'r+' or 'l+', plus a filler '+'
            move = [['d+', 'ul+'], ['d+', 'u+'], ['d+', 'ur+']][np.random.randint(3)]
            return move
        # Otherwise choose a repeated direction
        jump = np.random.choice(['ul+', 'u+', 'ur+'])
        return [jump]

    def string_to_idx(self, string_list: list):
        idx_list = []
        for s in string_list:
            try:
                idx_list.append(self.action_idx_lookup[s])
            except KeyError:
                # If something fails, we skip
                print(f"{s} not found in action lookup")
                print(string_list)
                idx_list.append(np.random.randint(0, len(self.action_idx_lookup)))
        return idx_list

    def sample(self,
               prob_jump: float = 0.04,
               prob_basic: float = 0.21,
               prob_combo: float = 0.35,
               prob_cancel: float = 0.2,
               prob_movement: float = 0.3):
        """
        Main entry point to sample an action for single or vs mode.
        In single mode, returns a single integer (action index).
        In vs mode, returns a tuple of two integers for agent_0, agent_1.

        The method picks from:
          - jump / overhead moves,
          - basic single-step actions,
          - combos or special sequences,
          - movement actions

        Probabilities can be configured by the user.

        Parameters
        ----------
        combo : bool, optional
            Whether combos are allowed in sampling (default True).
        prob_jump : float, optional
            Probability to select a "jump" action or overhead action. (default 0.04)
        prob_basic : float, optional
            Probability to pick a single random basic action (default 0.25)
        prob_combos : float, optional
            Probability to sample a special combo (default 0.70)
        prob_cancel : float, optional
            Probability that a combo will terminate early (default 0.20)
        prob_movement : float, optional
            Probability to do movement-based combos (default 0.40)
        vs : bool, optional
            Whether we are sampling for two players (default False).

        Returns
        -------
        int or (int, int)
            If vs=False, returns a single integer action.
            If vs=True, returns a tuple (action_0, action_1).

        Notes
        -----
        - This snippet references `self.sample_character_special(...)` and
          `self.sample_movement_action()`. Ensure you have a way to pick the correct
          character & super art for each agent if needed.
        """
        actions = {'discrete' : {},
                   'multi_discrete' : {}}
        probs = np.array([prob_jump, prob_basic, prob_combo, prob_movement])
        probs = probs / probs.sum()
        probs = np.cumsum(probs)
        for agent_id in self.agent_state:
            if not self.in_sequence(agent_id):
                roll = np.random.rand()
                if roll < probs[0]:
                    # Jump or overhead action placeholder
                    seq_str = self.sample_jump_action()
                    self.agent_state[agent_id]['move_sequence'] = deque(self.string_to_idx(seq_str))
                elif roll < probs[1]:
                    # Single basic action
                    single_idx = np.random.randint(0, len(self.action_idx_lookup))
                    self.agent_state[agent_id]['move_sequence'] = deque([single_idx])
                elif roll < probs[2]:
                    seq_str = self.sample_character_special(agent_id)
                    # Convert string combos to indices if possible
                    seq_idx = self.string_to_idx(seq_str)
                    cancel_roll = np.random.rand()
                    # Possibly shorten the sequence
                    if cancel_roll < prob_cancel:
                        cutoff = np.random.randint(1, len(seq_idx)+1)
                        seq_idx = seq_idx[:cutoff]
                    self.agent_state[agent_id]['move_sequence'] = deque(seq_idx)
                else:
                    # Movement-based combos
                    move_seq_str = self.sample_movement_action()
                    # Convert to indices
                    seq_idx = self.string_to_idx(move_seq_str)
                    self.agent_state[agent_id]['move_sequence'] = deque(seq_idx)
                    
            a_idx = self.agent_state[agent_id]['move_sequence'].popleft()
            a_m_d = self.input_lookup[a_idx]
            
            actions['discrete'][agent_id] = a_idx
            actions['multi_discrete'][agent_id] = a_m_d
        
        return actions
    