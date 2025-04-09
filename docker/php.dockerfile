FROM php:8.2-fpm

# Install system dependencies and required packages
RUN apt-get update && apt-get install -y \
    libmagickwand-dev \
    unzip \
    curl \
    libpng-dev \
    libonig-dev \
    libxml2-dev \
    zip \
    default-libmysqlclient-dev \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Install Imagick via PECL and enable it
RUN pecl install imagick && docker-php-ext-enable imagick

RUN sed -i 's/<policy domain="coder" rights="none" pattern="PDF" \/>/<policy domain="coder" rights="read|write" pattern="PDF" \/>/g' /etc/ImageMagick-6/policy.xml

# Install PHP extensions (including pdo_mysql)
RUN docker-php-ext-install pdo_mysql mbstring exif pcntl bcmath gd



WORKDIR /var/www

# Install Composer from the official Composer image
COPY --from=composer:latest /usr/bin/composer /usr/bin/composer

# Copy your Laravel application files
COPY . .

# (Optional) Force-enable pdo_mysql configuration file after copying project files
# RUN echo "extension=pdo_mysql.so" > /usr/local/etc/php/conf.d/99-pdo_mysql.ini

RUN chown -R www-data:www-data /var/www




CMD ["php-fpm"]
