import pygame
import sys
from chess_board import ChessBoard, Position, PieceType, PieceColor, Piece, Move, BOARD_SIZE, SQUARE_SIZE, WINDOW_SIZE
from interface import ChessAIManager

class ChessGame:
    def __init__(self):
        # Initialize pygame
        pygame.init()
        
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Chess Game")
        self.clock = pygame.time.Clock()
        self.board = ChessBoard()
        self.selected_position = None
        self.images = {}
        self.load_images()
        self.game_over = False
        self.result_message = ""
        
        # Initialize AI
        self.ai_manager = ChessAIManager(self)
        self.ai_enabled = False
    
    def load_images(self):
        """Load or create placeholder images for chess pieces"""
        piece_types = [p.value for p in PieceType if p != PieceType.EMPTY]
        colors = [c.value for c in PieceColor if c != PieceColor.NONE]
        
        for piece_type in piece_types:
            for color in colors:
                image_path = f"assets/{color}_{piece_type}.png"
                try:
                    self.images[f"{color}_{piece_type}"] = pygame.transform.scale(
                        pygame.image.load(image_path),
                        (SQUARE_SIZE, SQUARE_SIZE)
                    )
                except pygame.error:
                    # Create a placeholder image with text
                    img = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                    font = pygame.font.SysFont("Arial", 20)
                    text = font.render(f"{color[0]}{piece_type[0]}", True, (0, 0, 0))
                    img.blit(text, text.get_rect(center=(SQUARE_SIZE/2, SQUARE_SIZE/2)))
                    self.images[f"{color}_{piece_type}"] = img
    
    def draw_board(self):
        """Draw the chess board with pieces and highlights"""
        # Draw the board squares
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                # Determine square color (light or dark)
                color = (240, 217, 181) if (row + col) % 2 == 0 else (181, 136, 99)
                
                # Highlight selected piece
                if self.selected_position and row == self.selected_position.row and col == self.selected_position.col:
                    color = (124, 192, 214)  # Highlight selected
                
                # Draw square
                pygame.draw.rect(
                    self.screen, 
                    color, 
                    pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                )
        
        # Highlight valid moves for selected piece
        if self.selected_position:
            self.highlight_valid_moves()
        
        # Draw pieces
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.board.board[row][col]
                if piece.type != PieceType.EMPTY:
                    piece_img = self.images.get(f"{piece.color.value}_{piece.type.value}")
                    if piece_img:
                        self.screen.blit(
                            piece_img, 
                            pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                        )
        
        # Draw game state indicators
        self.draw_game_state_indicators()
        
        # Draw game over message if applicable
        if self.game_over:
            self.draw_game_over_message()
    
    def highlight_valid_moves(self):
        """Highlight valid moves for the selected piece"""
        if not self.selected_position:
            return
            
        valid_moves = self.board.get_valid_moves(self.selected_position)
        
        for move in valid_moves:
            end_row, end_col = move.end_pos.row, move.end_pos.col
            
            # Different highlight colors based on move type
            if move.captured_piece and move.captured_piece.type != PieceType.EMPTY:
                # Capture move - red highlight
                color = (255, 100, 100, 150)  # Red with alpha
            elif move.is_castling:
                # Castling move - purple highlight
                color = (180, 100, 255, 150)  # Purple with alpha
            elif move.is_en_passant:
                # En passant move - orange highlight
                color = (255, 165, 0, 150)  # Orange with alpha
            else:
                # Regular move - green highlight
                color = (100, 255, 100, 150)  # Green with alpha
                
            # Create a transparent surface for the highlight
            s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(s, color, s.get_rect())
            self.screen.blit(s, (end_col * SQUARE_SIZE, end_row * SQUARE_SIZE))
    
    def draw_game_state_indicators(self):
        """Draw indicators for game state (check, moves since capture, etc.)"""
        font = pygame.font.SysFont('Arial', 20)
        
        # Create indicator texts
        texts = []
        
        # Current player's turn
        texts.append(f"Turn: {self.board.turn.value.capitalize()}")
        
        # AI status
        if self.ai_enabled:
            white_ai = "AI" if self.ai_manager.white_ai else "Human"
            black_ai = "AI" if self.ai_manager.black_ai else "Human"
            texts.append(f"Players: White: {white_ai}, Black: {black_ai}")
        
        # Check indicator
        if self.board.is_check(self.board.turn):
            texts.append(f"{self.board.turn.value.capitalize()} is in CHECK!")
        
        # Fifty-move rule counter
        half_moves = self.board.half_move_clock
        texts.append(f"Moves since pawn/capture: {half_moves // 2}")
        
        # Display texts at the bottom of the window
        for i, text in enumerate(texts):
            text_surface = font.render(text, True, (255, 255, 255))
            text_rect = text_surface.get_rect()
            text_rect.bottomleft = (10, WINDOW_SIZE - 10 - i * 30)
            
            # Add a semi-transparent background
            bg_rect = text_rect.copy()
            bg_rect.inflate_ip(20, 10)
            s = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
            pygame.draw.rect(s, (0, 0, 0, 180), s.get_rect(), border_radius=5)
            self.screen.blit(s, bg_rect)
            self.screen.blit(text_surface, text_rect)
    
    def draw_game_over_message(self):
        """Draw the game over message"""
        # Create a semi-transparent overlay
        overlay = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))  # Black with alpha
        self.screen.blit(overlay, (0, 0))
        
        # Create message box
        font = pygame.font.SysFont('Arial', 36)
        text_surface = font.render(self.result_message, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2))
        
        # Create border around text
        border_rect = text_rect.copy()
        border_rect.inflate_ip(40, 40)
        pygame.draw.rect(self.screen, (255, 255, 255), border_rect, border_radius=10)
        pygame.draw.rect(self.screen, (0, 0, 0), border_rect.inflate(-6, -6), border_radius=8)
        
        # Draw text
        self.screen.blit(text_surface, text_rect)
        
        # Add instructions to restart
        restart_font = pygame.font.SysFont('Arial', 24)
        restart_text = restart_font.render("Press R to restart game", True, (200, 200, 200))
        restart_rect = restart_text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2 + 50))
        self.screen.blit(restart_text, restart_rect)
    
    def handle_click(self, pos):
        """Handle mouse clicks on the board"""
        if self.game_over:
            return
            
        # If AI is active for current player, ignore clicks
        if self.ai_enabled:
            current_color = self.board.turn
            if (current_color == PieceColor.WHITE and self.ai_manager.white_ai) or \
               (current_color == PieceColor.BLACK and self.ai_manager.black_ai):
                return
            
        col = pos[0] // SQUARE_SIZE
        row = pos[1] // SQUARE_SIZE
        
        clicked_position = Position(row, col)
        clicked_piece = self.board.get_piece(clicked_position)
        
        if self.selected_position:
            # If a piece is already selected, try to move it
            if clicked_position.row == self.selected_position.row and clicked_position.col == self.selected_position.col:
                # Deselect if clicking the same position
                self.selected_position = None
            else:
                # Try to move selected piece to the clicked position
                move = self.board.move_piece(self.selected_position, clicked_position)
                if move:
                    # Check game state after move
                    self.check_game_state()
                self.selected_position = None
        else:
            # Select a piece if it belongs to the current player
            if clicked_piece.type != PieceType.EMPTY and clicked_piece.color == self.board.turn:
                self.selected_position = clicked_position
    
    def check_game_state(self):
        """Check for checkmate, stalemate, and draws after a move"""
        opponent_color = PieceColor.BLACK if self.board.turn == PieceColor.WHITE else PieceColor.WHITE
        
        if self.board.is_checkmate(self.board.turn):
            self.game_over = True
            self.result_message = f"Checkmate! {opponent_color.value.capitalize()} wins!"
            self.ai_manager.stop_ai_game()
        elif self.board.is_stalemate(self.board.turn):
            self.game_over = True
            self.result_message = "Stalemate! Draw."
            self.ai_manager.stop_ai_game()
        elif self.board.is_fifty_move_rule_draw():
            self.game_over = True
            self.result_message = "Draw by fifty-move rule."
            self.ai_manager.stop_ai_game()
        elif self.board.is_threefold_repetition():
            self.game_over = True
            self.result_message = "Draw by threefold repetition."
            self.ai_manager.stop_ai_game()
    
    def reset_game(self):
        """Reset the game to the initial state"""
        self.board = ChessBoard()
        self.selected_position = None
        self.game_over = False
        self.result_message = ""
        if self.ai_enabled:
            # Preserve current AI settings
            white_ai = self.ai_manager.white_ai
            black_ai = self.ai_manager.black_ai
            self.ai_manager = ChessAIManager(self)
            if white_ai:
                self.ai_manager.register_ai(white_ai, PieceColor.WHITE)
            if black_ai:
                self.ai_manager.register_ai(black_ai, PieceColor.BLACK)
            self.ai_manager.start_ai_game()
    
    def toggle_ai(self, white_ai=None, black_ai=None):
        """Toggle AI players"""
        if white_ai:
            self.ai_manager.register_ai(white_ai, PieceColor.WHITE)
        if black_ai:
            self.ai_manager.register_ai(black_ai, PieceColor.BLACK)
        
        if white_ai or black_ai:
            self.ai_enabled = True
            self.ai_manager.start_ai_game()
        else:
            self.ai_enabled = False
            self.ai_manager.stop_ai_game()
    
    def run(self):
        """Main game loop"""
        running = True
        while running:
            current_time = pygame.time.get_ticks()
            
            # Update AI if enabled
            if self.ai_enabled:
                self.ai_manager.update(current_time)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r and self.game_over:  # Restart game with R key
                        self.reset_game()
                    elif event.key == pygame.K_1:  # Toggle AI playing as white
                        self.toggle_ai(white_ai=RandomChessAI())
                    elif event.key == pygame.K_2:  # Toggle AI playing as black
                        self.toggle_ai(black_ai=RandomChessAI())
                    elif event.key == pygame.K_3:  # AI vs AI
                        self.toggle_ai(white_ai=RandomChessAI(), black_ai=RandomChessAI())
                    elif event.key == pygame.K_0:  # Human vs Human
                        self.toggle_ai()
            
            # Clear the screen
            self.screen.fill((0, 0, 0))
            
            # Draw the board with all visual indicators
            self.draw_board()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = ChessGame()
    game.run()